# -*- coding: utf-8 -*-
"""Stage 2 特征提取：冻结 backbone（FECNet），逐窗口提取 1080 维 mean||max 特征

对每个窗口保存三样东西（供下游训练/探测/评估，无需再跑 backbone）：
  feat   [N, 1080]  classifier 输入特征（Stage 2 attention 的输入）
  logits [N, 101]   Stage 1 单窗口分类 logits（投票基线用）
  label  [N] / mark [N]

特征通过 forward hook 取 model.classifier 的输入（即 mean||max 拼接后的
1080 维特征），backbone 全程 eval() + no_grad，与官方冻结训练口径一致。
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from stage2_modules import load_backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default=os.path.join(ROOT_DIR, 'ncaltech101_official_gen/'),
                   help='数据目录（含 train.h5/test.h5）')
    p.add_argument('--ckpt', default=os.path.join(
        ROOT_DIR, 'archive/old_checkpoints/checkpoint_official/best_checkpoint.pth'),
        help='Stage 1 backbone checkpoint（官方或自训）')
    p.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    p.add_argument('--out_dir', default=os.path.join(BASE_DIR, 'features'))
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num_point', type=int, default=10240)
    p.add_argument('--num_classes', type=int, default=101)
    p.add_argument('--limit', type=int, default=0, help='只处理前 N 个窗口（调试用，0=全部）')
    return p.parse_args()


def extract_split(model, device, h5_path, out_path, args, split_seed):
    # Codex 修复（2026-10-16）：每个 split 固定重播种——FPS 消耗 RNG，
    # 原版全局 torch.manual_seed(42) 只设一次，--split both 时 test 先跑、
    # train 后跑，--split train 时 train 从不同 RNG 状态开始 → train 特征
    # 依赖是否先抽了 test。这里每个 split 用固定 seed（test=42, train=43），
    # 保证特征与调用顺序无关、跨 run 可复现。
    torch.manual_seed(split_seed)
    with h5py.File(h5_path, 'r') as f:
        n_total = f['data'].shape[0]
        n = min(n_total, args.limit) if args.limit else n_total
        print(f'[{os.path.basename(h5_path)}] 共 {n_total} 窗口, 提取 {n}')

        # hook：抓 classifier 输入特征（1080 维）与输出 logits（101 维）
        buf = {}

        def hook(m, inp, out):
            buf['feat'] = inp[0].detach()

        h = model.classifier.register_forward_hook(hook)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, 'w') as out:
            d_feat = out.create_dataset('feat', (n, 1080), dtype='f4')
            d_logits = out.create_dataset('logits', (n, args.num_classes), dtype='f4')
            d_label = out.create_dataset('label', (n,), dtype='i8')
            d_mark = out.create_dataset('mark', (n,), dtype='i8')

            t0 = time.time()
            with torch.no_grad():
                for i in range(0, n, args.batch):
                    j = min(i + args.batch, n)
                    data = np.array(f['data'][i:j], dtype=np.float32)  # [b, N, 4]
                    label = np.array(f['label'][i:j])
                    mark = np.array(f['mark'][i:j])
                    x = torch.from_numpy(data).permute(0, 2, 1).to(device)  # [b, 4, N]
                    logits = model(x)
                    d_feat[i:j] = buf['feat'].cpu().numpy()
                    d_logits[i:j] = logits.cpu().numpy()
                    d_label[i:j] = label
                    d_mark[i:j] = mark
                    if (i // args.batch) % 50 == 0:
                        speed = j / max(time.time() - t0, 1e-6)
                        eta = (n - j) / max(speed, 1e-6) / 60
                        print(f'  {j}/{n} ({speed:.1f} win/s, ETA {eta:.1f} min)', flush=True)
        h.remove()
    print(f'  完成, 保存到 {out_path}, 耗时 {(time.time()-t0)/60:.1f} min')


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    # Codex 建议（2026-10-16）：num_classes 从 ckpt.args.num_category 读
    # （覆盖默认 101，兼容未来 5 类 sanity）。load_backbone 前先 peek ckpt。
    _ckpt = torch.load(args.ckpt, map_location='cpu')
    _num_cat = (_ckpt.get('args', {}) or {}).get('num_category')
    if _num_cat is not None:
        args.num_classes = int(_num_cat)
        print(f'[num_classes] 从 ckpt.args.num_category 读 = {args.num_classes}')

    model = load_backbone(args.ckpt, num_classes=args.num_classes)
    model.to(device).eval()  # 冻结：eval 模式，BN 用 running stats，无 dropout

    # split_seed：test=42, train=43（与调用顺序无关的固定值）
    if args.split in ('test', 'both'):
        extract_split(model, device,
                      os.path.join(args.data_path, 'test.h5'),
                      os.path.join(args.out_dir, 'features_test.h5'), args, 42)
    if args.split in ('train', 'both'):
        extract_split(model, device,
                      os.path.join(args.data_path, 'train.h5'),
                      os.path.join(args.out_dir, 'features_train.h5'), args, 43)


if __name__ == '__main__':
    main()
