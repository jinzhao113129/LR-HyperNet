# -*- coding: utf-8 -*-
"""Stage 2 评估：按 mark 分组测试窗口 → 聚合 → 视频级 top-1 准确率

同时输出 Stage 1 基线（单窗口 / 多数投票 / 概率平均 / 概率最大），
便于对照"跨窗口聚合带来的提升"。
支持加载自训聚合器 ckpt 或官方 Stage 2 权重。
"""
import argparse
import os
import sys
from collections import Counter

import h5py
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from stage2_modules import WindowAggregator, load_official_stage2_state

OFFICIAL_CKPT = os.path.join(
    ROOT_DIR, 'archive/old_checkpoints/checkpoint_official/best_checkpoint.pth')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default=os.path.join(BASE_DIR, 'features/features_test.h5'))
    p.add_argument('--ckpt', default=os.path.join(BASE_DIR, 'log_stage2/best_checkpoint.pth'),
                   help='自训聚合器 ckpt 目录或 .pth 路径；--ckpt official 则用官方权重')
    p.add_argument('--which', default='both', choices=['last', 'best', 'both'],
                   help='Codex P0 修复：last=主口径（干净，训练终点无选择偏置）；'
                        'best=exploratory（test-select 偏置，Stage-2 按测试集选 best）；'
                        'both=两个都评（默认）')
    p.add_argument('--official', action='store_true', help='直接评估官方 Stage 2 权重')
    p.add_argument('--batch', type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with h5py.File(args.features, 'r') as f:
        feat = np.array(f['feat'][:], dtype=np.float32)
        logits = np.array(f['logits'][:], dtype=np.float32)
        label = np.array(f['label'][:]).astype(np.int64)
        mark = np.array(f['mark'][:]).astype(np.int64)

    # 按 mark 分组（组内按出现顺序 = 时间窗顺序）
    groups, g_labels, g_logits = [], [], []
    seen = {}
    for i, m in enumerate(mark):
        seen.setdefault(int(m), []).append(i)
    for m in sorted(seen.keys()):
        idxs = seen[m]
        assert len(idxs) == 3, f'mark={m} 窗口数 {len(idxs)} != 3'
        groups.append(feat[idxs])
        g_labels.append(label[idxs[0]])
        g_logits.append(logits[idxs])
    x = torch.from_numpy(np.stack(groups)).to(device)      # [n_video, 3, 1080]
    y = torch.from_numpy(np.array(g_labels)).to(device)
    g_logits = np.stack(g_logits)                          # [n_video, 3, 101]
    n_video = len(y)
    print(f'视频数: {n_video}')
    # Codex 第三轮非阻断#2：口径诚实声明——Stage-1 特征来自 175245 的
    # best_checkpoint（其 best 本身按测试集 seq 选，带 test-select 偏置），
    # 故最终口径 = "Stage-1 best-on-test + Stage-2 last/val-selected"，
    # 不能笼统称完全干净。论文若主张完全干净需补 Stage-1 last 对照矩阵。
    print('[口径] Stage-1 特征 = 175245 best(自身 test-select)；'
          'Stage-2 last 消除 Stage-2 选择偏置，但 Stage-1 的 test-select 仍在')

    # ---------- Stage 1 基线 ----------
    win_pred_all = logits.argmax(1)
    print(f'[基线] 单窗口 top-1: {(win_pred_all == label).mean()*100:.3f}%')
    probs = torch.softmax(torch.from_numpy(g_logits), dim=2).numpy()
    preds = g_logits.argmax(2)
    a_major = a_pmean = a_pmax = 0
    for v in range(n_video):
        a_major += (Counter(preds[v].tolist()).most_common(1)[0][0] == g_labels[v])
        a_pmean += (probs[v].mean(0).argmax() == g_labels[v])
        a_pmax += (probs[v][probs[v].max(1).argmax()].argmax() == g_labels[v])
    print(f'[基线] 多数投票:     {a_major/n_video*100:.3f}%')
    print(f'[基线] 概率平均:     {a_pmean/n_video*100:.3f}%')
    print(f'[基线] 概率最大:     {a_pmax/n_video*100:.3f}%')

    # ---------- Stage 2 聚合 ----------
    if args.official:
        ref = load_official_stage2_state(OFFICIAL_CKPT)
        modes = ['attn_concat', 'raw_concat', 'wsum_tile', 'gate_concat', 'wsum_meanmax']
        print('\n官方权重 × 各候选前向:')
        for mode in modes:
            agg = WindowAggregator(forward_mode=mode)
            agg.load_state_dict(ref, strict=True)
            agg.to(device).eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, n_video, args.batch):
                    correct += (agg(x[i:i+args.batch]).argmax(1)
                                == y[i:i+args.batch]).sum().item()
            print(f'  {mode:<16}{correct/n_video*100:.3f}%')
        return

    # Codex P0 修复（2026-10-16）：支持 --which both 双口径评估。
    # Stage-1 的 175245 best 本身按测试集 seq 选出，Stage-2 的 best 又按
    # 测试集视频 acc 选出——双重 best-on-test 会污染"对 82.4"的终局结论。
    # 口径约定：last 为主口径（干净：训练终点，无测试集选择偏置）；
    # best 为 exploratory（标注 test-select 偏置），两者都报。
    which_list = [args.which] if args.which != 'both' else ['last', 'best']
    for which in which_list:
        ckpt_name = 'last_checkpoint.pth' if which == 'last' else 'best_checkpoint.pth'
        ckpt_path = os.path.join(os.path.dirname(args.ckpt), ckpt_name) \
            if args.ckpt.endswith('.pth') else os.path.join(args.ckpt, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f'\n[Stage 2] {which}: 无 {ckpt_name}，跳过')
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg = ckpt.get('args', {})
        mode = cfg.get('forward_mode', 'attn_concat')
        agg = WindowAggregator(forward_mode=mode)
        agg.load_state_dict(ckpt['net'], strict=True)
        agg.to(device).eval()
        # Codex 第三轮非阻断#1 修复：tag 读 checkpoint 后动态设置——
        # last 恒为主口径（干净）；best 按 checkpoint 存的键判定
        # val_acc → val-selected（干净）；test_acc → test-selected(exploratory)。
        # 不再预先统一标 exploratory（避免与细节行矛盾）。
        if which == 'last':
            tag = '主口径(干净)'
        elif 'val_acc' in ckpt:
            tag = 'val-selected(干净)'
        else:
            tag = 'test-selected(exploratory)'
        print(f'\n[Stage 2] {which} ({tag}): epoch {ckpt.get("epoch")}, '
              f'forward_mode={mode}')
        correct = 0
        with torch.no_grad():
            for i in range(0, n_video, args.batch):
                correct += (agg(x[i:i+args.batch]).argmax(1)
                            == y[i:i+args.batch]).sum().item()
        acc = correct / n_video * 100
        print(f'[Stage 2] {which} 视频级 acc: {acc:.3f}%  ({correct}/{n_video})')
        if which == 'best':
            if 'val_acc' in ckpt:
                print(f'          训练时选择口径: val-selected(干净) '
                      f'val_acc={ckpt.get("val_acc")}%')
            else:
                print(f'          训练时选择口径: test-selected(exploratory) '
                      f'test_acc={ckpt.get("test_acc", "n/a")}%')
        print(f'          官方参考: 82.377%')


if __name__ == '__main__':
    main()
