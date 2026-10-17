# -*- coding: utf-8 -*-
"""Stage 2 训练：冻结 backbone 特征上训练跨窗口聚合器（复现官方配方）

训练配方依据（官方 checkpoint optimizer 状态逆向）：
  - 优化器 SGD, momentum=0.9, 初始 lr=0.1, weight_decay=1e-4
    （保存时 lr 已衰减到 0.0572，呈余弦衰减中段形态，故默认余弦退火）
  - 仅 12 个参数张量有动量缓冲（attention 2 + classifier Linear 6 + BN 4），
    backbone 93 个参数无梯度 → 冻结 backbone、只训聚合器
  - best_train_acc=99.986 → 训练到收敛

训练样本构造：train.h5 每视频 9 窗口 = 3 个时间窗 × 3 种变体
[原始, 随机裁剪, 时间反转]（generate_ncaltech101_hpc.py 的生成顺序）。
一个 Stage 2 样本 = 每个时间窗各取一个变体：
  --sample_mode orig    只用原始窗口 [0,3,6]（6967 样本/epoch）
  --sample_mode triple  三种变体各成一个样本 [0,3,6],[1,4,7],[2,5,8]
                        （20901 样本/epoch，相当于变体级数据增强）
  --sample_mode random  每 epoch 每视频随机抽一种变体组合（在线增强）
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from stage2_modules import WindowAggregator, load_official_stage2_state

OFFICIAL_CKPT = os.path.join(
    ROOT_DIR, 'archive/old_checkpoints/checkpoint_official/best_checkpoint.pth')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_features', default=os.path.join(BASE_DIR, 'features/features_train.h5'))
    p.add_argument('--test_features', default=os.path.join(BASE_DIR, 'features/features_test.h5'))
    p.add_argument('--out_dir', default=os.path.join(BASE_DIR, 'log_stage2'))
    p.add_argument('--forward_mode', default='attn_concat',
                   choices=['attn_concat', 'raw_concat', 'wsum_tile',
                            'gate_concat', 'wsum_meanmax'])
    p.add_argument('--sample_mode', default='triple',
                   choices=['orig', 'triple', 'random'])
    p.add_argument('--init', default='scratch', choices=['scratch', 'official'],
                   help='scratch=随机初始化（复现训练过程）；official=官方权重续训/微调')
    p.add_argument('--init_ckpt', default=OFFICIAL_CKPT,
                   help='--init official 时的权重来源')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1, help='官方 optimizer: 初始 lr=0.1')
    p.add_argument('--wd', type=float, default=1e-4, help='官方 weight_decay=1e-4')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'step', 'none'])
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--num_windows', type=int, default=3)
    p.add_argument('--feat_dim', type=int, default=1080)
    p.add_argument('--num_classes', type=int, default=101)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--val_ratio', type=float, default=0.0,
                   help='Codex P0 修复：>0 时从 train 特征按视频划分验证集'
                        '（如 0.1=10%% 视频作验证），用验证集选 best（干净的'
                        '模型选择），测试集只在训练后由 stage2_eval.py 评一次；'
                        '=0 保持旧行为（测试集选 best，带 test-select 偏置）')
    return p.parse_args()


def load_features(path):
    with h5py.File(path, 'r') as f:
        feat = np.array(f['feat'][:], dtype=np.float32)
        label = np.array(f['label'][:]).astype(np.int64)
        mark = np.array(f['mark'][:]).astype(np.int64)
    return feat, label, mark


def group_indices(mark):
    """mark -> 该视频全部窗口的 h5 行号（按出现顺序）"""
    groups = {}
    for i, m in enumerate(mark):
        groups.setdefault(int(m), []).append(i)
    return groups


class WindowTripleDataset(Dataset):
    """把每视频的窗口打包成 (3, 1080) 样本。

    每视频 9 窗口布局：[w1:orig,crop,rev][w2:orig,crop,rev][w3:orig,crop,rev]
    样本 = 每个时间窗各取同一变体序号 v：行号 [0*3+v, 1*3+v, 2*3+v]。
    """

    def __init__(self, feat, label, mark, sample_mode='triple', num_windows=3):
        self.feat = torch.from_numpy(feat)
        self.groups = group_indices(mark)
        self.video_ids = sorted(self.groups.keys())
        self.video_label = {m: torch.tensor(label[idxs[0]], dtype=torch.long)
                            for m, idxs in self.groups.items()}
        self.sample_mode = sample_mode
        self.num_windows = num_windows
        # 每视频的变体数（窗口数/时间窗数，正常 = 3）
        self.n_variant = {m: len(idxs) // num_windows
                          for m, idxs in self.groups.items()}
        # 预生成样本列表（triple/orig 模式固定；random 模式每 epoch重抽）
        self.samples = []
        if sample_mode != 'random':
            for m in self.video_ids:
                nv = self.n_variant[m]
                vs = [0] if sample_mode == 'orig' else range(nv)
                for v in vs:
                    self.samples.append((m, v))

    def refresh_random(self, epoch):
        """random 模式：每 epoch 每视频随机抽一个变体组合"""
        rng = np.random.default_rng(self.seed0 + epoch)
        self.samples = []
        for m in self.video_ids:
            v = int(rng.integers(0, max(self.n_variant[m], 1)))
            self.samples.append((m, v))

    seed0 = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        m, v = self.samples[i]
        idxs = self.groups[m]
        # 时间窗 w 的第 v 个变体在 h5 中的行号 = w*3+v
        rows = [idxs[w * self.n_variant[m] + v] for w in range(self.num_windows)]
        return self.feat[rows], self.video_label[m]


def build_orig_window_tensor(feat, label, mark, num_windows=3):
    """训练特征（每视频 9 窗 = 3 时间窗 × 3 变体）打包成 [n_video, W, D]。

    Codex P0 修复（2026-10-16）：验证集从 train 特征划分而来，每视频
    9 窗（orig/crop/rev 变体），不是测试集的恰好 3 原始窗——直接用
    build_test_tensor 会触发 `assert len(idxs)==3` 崩溃。此处取每视频
    的**原始三个时间窗**（每时间窗的第 0 个变体，即 rows = idxs[::3]
    等价的 w*n_variant+0 采样），不混入 crop/reverse 变体。
    布局约定与 WindowTripleDataset 一致：行号 = w*n_variant + v，
    原始窗 = v 0。
    """
    groups = group_indices(mark)
    xs, ys = [], []
    for m in sorted(groups.keys()):
        idxs = groups[m]
        n_variant = len(idxs) // num_windows
        # 原始时间窗：每个时间窗的第 0 个变体
        rows = [idxs[w * n_variant] for w in range(num_windows)]
        xs.append(torch.from_numpy(feat[rows]))
        ys.append(label[idxs[0]])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def build_test_tensor(feat, label, mark, num_windows=3):
    """测试集打包成 [n_video, W, D]（测试每视频恰好 W 个原始窗口）"""
    groups = group_indices(mark)
    xs, ys = [], []
    for m in sorted(groups.keys()):
        idxs = groups[m]
        assert len(idxs) == num_windows
        xs.append(torch.from_numpy(feat[idxs]))
        ys.append(label[idxs[0]])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def evaluate(agg, x, y, batch=512):
    agg.eval()
    correct = 0
    for i in range(0, len(x), batch):
        out = agg(x[i:i+batch])
        correct += (out.argmax(1) == y[i:i+batch]).sum().item()
    return correct / len(x) * 100


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'device: {device} | mode: {args.forward_mode} | sample: {args.sample_mode}')
    print(args)

    # ---------- 数据 ----------
    feat_tr, lab_tr, mark_tr = load_features(args.train_features)
    feat_te, lab_te, mark_te = load_features(args.test_features)
    print(f'train 特征 {feat_tr.shape} ({len(np.unique(mark_tr))} 视频), '
          f'test 特征 {feat_te.shape} ({len(np.unique(mark_te))} 视频)')

    # ---------- Codex P0 修复：按视频划分 Stage-2 验证集（可选）----------
    # val_ratio > 0 时：从 train 特征按视频抽 val_ratio 划出验证集，
    # 用验证集选 best_checkpoint（干净的模型选择），测试集只在训练后
    # 由 stage2_eval.py 评一次。val_ratio=0 保持旧行为（测试集选 best，
    # 输出目录里 best 标注 test-select 偏置）。
    val_feat = val_lab = val_mark = None
    if args.val_ratio > 0:
        video_ids = sorted(set(int(m) for m in mark_tr))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(video_ids)
        n_val = max(1, int(len(video_ids) * args.val_ratio))
        val_ids = set(video_ids[:n_val])
        is_val = np.array([int(m) in val_ids for m in mark_tr])
        val_feat, val_lab, val_mark = feat_tr[is_val], lab_tr[is_val], mark_tr[is_val]
        feat_tr, lab_tr, mark_tr = feat_tr[~is_val], lab_tr[~is_val], mark_tr[~is_val]
        print(f'[val split] train {feat_tr.shape[0]} 窗 ({len(set(int(m) for m in mark_tr))} 视频) | '
              f'val {val_feat.shape[0]} 窗 ({len(val_ids)} 视频) | ratio {args.val_ratio}')

    train_ds = WindowTripleDataset(feat_tr, lab_tr, mark_tr,
                                   sample_mode=args.sample_mode,
                                   num_windows=args.num_windows)
    train_ds.seed0 = args.seed * 1000
    trainloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                             num_workers=2, drop_last=False)
    # 模型选择用的评估集：val_ratio>0 → 验证集（干净，原始 3 时间窗，
    # 用 build_orig_window_tensor 从 9 窗取原始窗，不混变体）；否则
    # test（旧行为，stage2_eval.py 输出时该 best 标注 test-select 偏置）
    if args.val_ratio > 0:
        x_sel, y_sel = build_orig_window_tensor(val_feat, val_lab, val_mark,
                                                 args.num_windows)
        sel_name = 'val'
    else:
        x_sel, y_sel = build_test_tensor(feat_te, lab_te, mark_te,
                                         args.num_windows)
        sel_name = 'test'
    x_sel, y_sel = x_sel.to(device), y_sel.to(device)
    x_te, y_te = build_test_tensor(feat_te, lab_te, mark_te, args.num_windows)
    x_te, y_te = x_te.to(device), y_te.to(device)

    # ---------- 模型 ----------
    agg = WindowAggregator(feat_dim=args.feat_dim, num_windows=args.num_windows,
                           num_classes=args.num_classes,
                           forward_mode=args.forward_mode)
    if args.init == 'official':
        agg.load_state_dict(load_official_stage2_state(args.init_ckpt), strict=True)
        print('已加载官方 Stage 2 权重作为初始化')
    agg.to(device)

    # ---------- 优化器（官方配方） ----------
    optimizer = torch.optim.SGD(agg.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc, best_epoch = -1.0, -1
    log_path = os.path.join(args.out_dir, 'train_log.txt')
    for epoch in range(1, args.epochs + 1):
        if args.sample_mode == 'random':
            train_ds.refresh_random(epoch)
        agg.train()
        t0 = time.time()
        tot = correct = 0
        loss_sum = 0.0
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = agg(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        if scheduler is not None:
            scheduler.step()

        msg = (f'epoch {epoch:3d} | lr {optimizer.param_groups[0]["lr"]:.5f} | '
               f'loss {loss_sum/tot:.4f} | train acc {correct/tot*100:.3f}%')
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc = evaluate(agg, x_sel, y_sel)
            msg += f' | {sel_name} video acc {acc:.3f}%'
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                save_key = 'val_acc' if args.val_ratio > 0 else 'test_acc'
                torch.save({'net': agg.state_dict(), 'epoch': epoch,
                            save_key: acc, 'args': vars(args)},
                           os.path.join(args.out_dir, 'best_checkpoint.pth'))
                msg += ' | *best*'
        msg += f' | {time.time()-t0:.1f}s'
        print(msg, flush=True)
        with open(log_path, 'a') as flog:
            flog.write(msg + '\n')

    # 保存最终权重（对照用）
    torch.save({'net': agg.state_dict(), 'epoch': args.epochs,
                'args': vars(args)},
               os.path.join(args.out_dir, 'last_checkpoint.pth'))
    print(f'\n完成: best video acc {best_acc:.3f}% @ epoch {best_epoch} '
          f'(官方参考 82.377%), ckpt 在 {args.out_dir}')


if __name__ == '__main__':
    main()
