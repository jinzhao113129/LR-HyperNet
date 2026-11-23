"""可选三窗口Stage-2：显式提取/训练/评估；本地重建头，不声称官方精确复现。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import build_model, dump_json, seed_all, sha256
from data_io import EventH5, inspect_data, row_coordinates
from train_ddp import load_checkpoint


class ThreeWindowHead(nn.Module):
    """注意力加权后拼接三个1080维窗口，训练独立分类头。"""

    def __init__(self, dimension: int = 1080):
        """输入窗口维度，输出101类别，不改骨干。"""
        super().__init__()
        self.attention = nn.Linear(dimension, 1)
        self.classifier = nn.Sequential(nn.Dropout(.1), nn.Linear(3 * dimension, 1024),
                                        nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(.5),
                                        nn.Linear(1024, 1024), nn.BatchNorm1d(1024),
                                        nn.GELU(), nn.Dropout(.5), nn.Linear(1024, 101))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """将[B,3,1080]特征映射为[B,101]，保留窗口先后顺序。"""
        weights = self.attention(windows).softmax(dim=1)
        return self.classifier((windows * weights).flatten(1))


def backbone(path: Path) -> nn.Module:
    """只接受本包完成150轮的raw last模型，严格恢复门控。"""
    state = load_checkpoint(path)
    if path.name != 'last_checkpoint.pth' or state['epoch'] != 149:
        raise ValueError('Stage-2只接受完成150轮的last_checkpoint.pth')
    args = state['args']
    if not args['hyperbolic_gate'] or args['remote_g16'] or args['ema'] != 0:
        raise ValueError('不是本次单独双曲门控配方')
    model = build_model()
    model.load_state_dict(state['net'], strict=True)
    model.eval().requires_grad_(False)
    return model.cuda()


def extract(args: argparse.Namespace) -> None:
    """单卡提取全部窗口，保留mark/window/variant；不加载任何官方分类头。"""
    report = inspect_data(args.data)
    if args.out.exists():
        raise FileExistsError('特征输出必须为新目录')
    if not torch.cuda.is_available():
        raise RuntimeError('特征提取需要CUDA')
    model = backbone(args.stage1)
    metadata = json.loads((args.data / 'dataset_manifest.json').read_text(encoding='utf-8'))
    args.out.mkdir(parents=True)
    feature = {}
    hook = model.classifier.register_forward_pre_hook(lambda _, inp: feature.update(x=inp[0].detach()))
    fingerprint = sha256(args.stage1)
    for split, seed in [('train', 43), ('test', 42)]:
        seed_all(seed)
        data = EventH5(args.data / f'{split}.h5')
        loader = DataLoader(data, batch_size=args.batch, shuffle=False, num_workers=2)
        with h5py.File(args.data / f'{split}.h5', 'r') as source, h5py.File(args.out / f'{split}.h5', 'x') as out:
            for key in ('label', 'mark'):
                out[key] = source[key][:]
            windows, variants = row_coordinates(source, split, metadata)
            out['window'], out['variant'] = windows, variants
            target = out.create_dataset('feat', (len(data), 1080), dtype='f4')
            with torch.no_grad():
                for x, _, positions in loader:
                    model(x.cuda())
                    target[positions.numpy()] = feature['x'].cpu().numpy()
    hook.remove()
    if sha256(args.stage1) != fingerprint:
        raise ValueError('提取期间骨干权重改变')
    dump_json(args.out / 'source.json', dict(stage1_sha256=fingerprint, data=report,
              inference_batch=args.batch, seeds={'train': 43, 'test': 42}))


def grouped_features(path: Path, augment: bool) -> tuple:
    """按显式窗口元数据构造三窗，不依赖mark局部编号或H5行顺序。"""
    with h5py.File(path, 'r') as source:
        features, labels, marks = source['feat'][:], source['label'][:], source['mark'][:]
        windows, variants = source['window'][:], source['variant'][:]
    x, y, ids = [], [], []
    for mark in np.unique(marks):
        members = np.flatnonzero(marks == mark)
        for variant in (range(3) if augment else range(1)):
            lookup = {int(windows[i]): i for i in members if variants[i] == variant}
            if set(lookup) != {0, 1, 2}:
                raise ValueError(f'记录{mark}变体{variant}不完整，请回收skip清单后决定协议，不静默丢样本')
            rows = [lookup[i] for i in range(3)]
            if len(set(labels[rows].tolist())) != 1:
                raise ValueError('同记录类别不一致')
            x.append(features[rows])
            y.append(int(labels[rows[0]]))
            ids.append(int(mark))
    return torch.from_numpy(np.stack(x)), torch.tensor(y), np.asarray(ids)


def head_evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    """按记录计算准确率，不按窗口重复计数。"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for a, b in DataLoader(TensorDataset(x, y), batch_size=128):
            correct += int((model(a.cuda()).argmax(-1).cpu() == b).sum())
    return dict(correct=correct, total=len(y), accuracy=100 * correct / len(y))


def train_head(args: argparse.Namespace) -> None:
    """记录级10%验证，100轮SGD，只用训练记录更新头，不读取测试标签。"""
    args.out.mkdir(parents=True, exist_ok=False)
    x, y, ids = grouped_features(args.features / 'train.h5', True)
    order = np.random.RandomState(42).permutation(np.unique(ids))
    validation = set(order[:max(1, len(order) // 10)].tolist())
    mask = np.array([i in validation for i in ids])
    loader = DataLoader(TensorDataset(x[~mask], y[~mask]), batch_size=128, shuffle=True, drop_last=True)
    if len(loader) == 0 or mask.sum() == 0:
        raise ValueError('Stage-2训练/验证记录不足')
    model = ThreeWindowHead().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    dump_json(args.out / 'split.json', dict(validation_records=sorted(validation), seed=42,
              feature_source=sha256(args.features / 'source.json'), hypothesis='attn_concat_not_verified_official'))
    for epoch in range(100):
        model.train()
        for a, b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(a.cuda()), b.cuda())
            loss.backward()
            optimizer.step()
        scheduler.step()
        result = head_evaluate(model, x[mask], y[mask])
        print(json.dumps(dict(epoch=epoch + 1, validation=result)), flush=True)
        temporary = args.out / 'last_head.pth.tmp'
        torch.save(dict(net=model.state_dict(), epoch=epoch,
                        source_sha256=sha256(args.features / 'source.json')), temporary)
        temporary.replace(args.out / 'last_head.pth')


def main() -> None:
    """三个动作必须分别执行，不因Stage-1结束就自动进入Stage-2。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'train', 'evaluate'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--stage1', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--head', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    seed_all(42)
    if args.action == 'extract':
        if args.data is None or args.stage1 is None:
            parser.error('extract需要--data和--stage1')
        extract(args)
    elif args.action == 'train':
        if args.features is None:
            parser.error('train需要--features')
        train_head(args)
    else:
        if args.features is None or args.head is None:
            parser.error('evaluate需要--features和--head')
        args.out.mkdir(parents=True, exist_ok=False)
        state = load_checkpoint(args.head)
        if state['epoch'] != 99 or state['source_sha256'] != sha256(args.features / 'source.json'):
            raise ValueError('需要100轮last头且特征来源一致')
        model = ThreeWindowHead().cuda()
        model.load_state_dict(state['net'], strict=True)
        x, y, _ = grouped_features(args.features / 'test.h5', False)
        dump_json(args.out / 'test.json', head_evaluate(model, x, y))


if __name__ == '__main__':
    main()
