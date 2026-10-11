# -*- coding: utf-8 -*-
"""SECNet 基线跨窗口投票验证
对比四种口径：单窗口 top-1、多数投票、概率平均、概率最大
官方 82.4% 为"跨窗口投票"结果（README），本脚本验证我们模型投票后能到多少
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from FECNet import FECNet

DATA_PATH = './ncaltech101_official_gen/'
CKPT = './log/classification/secnet_official_gen_150ep_adamw_a800/checkpoints/last_checkpoint.pth'
BATCH = 16
NUM_POINT = 10240
NUM_CLASS = 101


def load_h5_mark(h5_filename):
    import h5py
    h5f = h5py.File(h5_filename, 'r')
    data = np.array(h5f['data'][:])
    label = np.array(h5f['label'][:])
    mark = np.array(h5f['mark'][:])
    h5f.close()
    return data, label, mark


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    data, label, mark = load_h5_mark(DATA_PATH + 'test.h5')
    label = label.astype('int64')
    data = data.reshape(-1, NUM_POINT, 4)
    print("test data:", data.shape, "label:", label.shape, "mark:", mark.shape)

    classifier = FECNet(num_classes=NUM_CLASS)
    checkpoint = torch.load(CKPT, map_location=device)
    classifier.load_state_dict(checkpoint['net'])
    classifier.to(device)
    classifier.eval()

    logits_all, label_all, mark_all = [], [], []
    with torch.no_grad():
        n = data.shape[0]
        for i in range(0, n, BATCH):
            xb = torch.from_numpy(data[i:i+BATCH]).float().permute(0, 2, 1).to(device)
            logits = classifier(xb)
            logits_all.append(logits.cpu())
            label_all.append(label[i:i+BATCH])
            mark_all.append(mark[i:i+BATCH])
    logits = torch.cat(logits_all, 0)   # [N, 101]
    labels = np.concatenate(label_all)
    marks = np.concatenate(mark_all)
    probs = torch.softmax(logits, dim=1).numpy()   # [N, 101]
    preds = logits.argmax(dim=1).numpy()

    # 1) 单窗口 top-1
    acc_win = (preds == labels).mean() * 100
    print(f"[1] 单窗口 top-1 acc: {acc_win:.3f}%")

    # 按 mark 分组
    groups = {}
    for idx, m in enumerate(marks):
        groups.setdefault(int(m), []).append(idx)
    n_video = len(groups)
    print(f"    视频数(分组数): {n_video}")

    cnt_major = cnt_prob_mean = cnt_prob_max = cnt_label_major = 0
    for m, idxs in groups.items():
        g_preds = preds[idxs]
        g_probs = probs[idxs]
        g_label = labels[idxs[0]]
        # 2) 多数投票
        from collections import Counter
        tar = Counter(g_preds.tolist()).most_common(1)[0][0]
        cnt_major += (tar == g_label)
        # 3) 概率平均
        tar_mean = g_probs.mean(axis=0).argmax()
        cnt_prob_mean += (tar_mean == g_label)
        # 4) 概率最大（取置信度最高的窗口）
        conf = g_probs.max(axis=1)
        tar_max = g_probs[conf.argmax()].argmax()
        cnt_prob_max += (tar_max == g_label)

    print(f"[2] 多数投票 acc:      {cnt_major/n_video*100:.3f}%  ({cnt_major}/{n_video})")
    print(f"[3] 概率平均投票 acc:  {cnt_prob_mean/n_video*100:.3f}%  ({cnt_prob_mean}/{n_video})")
    print(f"[4] 概率最大投票 acc:  {cnt_prob_max/n_video*100:.3f}%  ({cnt_prob_max}/{n_video})")
    print(f"    官方 82.4% 为跨窗口投票结果（README：先单窗口训练再简单分类器投票）")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"elapsed: {time.time()-t0:.1f}s")
