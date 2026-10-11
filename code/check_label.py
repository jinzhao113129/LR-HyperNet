# -*- coding: utf-8 -*-
"""检查 train.h5 / test.h5 中 label 与 mark 的取值分布，确认 label 是否为 0-100 连续类别。"""
import h5py
import numpy as np

for name in ['train', 'test']:
    path = f'/home/jnguan/JiangYouxin/ZhaoJin/data/{name}.h5'
    with h5py.File(path, 'r') as f:
        label = f['label'][:]
        mark = f['mark'][:]
        keys = list(f.keys())

    label = np.asarray(label).squeeze()

    uniq = np.unique(label)
    print(f'===== {name}.h5 =====')
    print(f'datasets: {keys}, label shape: {label.shape}, mark shape: {mark.shape}')
    print(f'label: min={label.min()}, max={label.max()}, unique={len(uniq)}')
    print(f'label 前 20 个唯一值: {uniq[:20].tolist()}')
    print(f'label 后 20 个唯一值: {uniq[-20:].tolist()}')

    mark = np.asarray(mark).squeeze()
    muniq = np.unique(mark)
    print(f'mark: min={mark.min()}, max={mark.max()}, unique={len(muniq)}')
    print(f'mark 前 10 个唯一值: {muniq[:10].tolist()}')
    print(f'mark 后 10 个唯一值: {muniq[-10:].tolist()}')

    print(f'前 5 个样本: label={label[:5].tolist()}, mark={np.asarray(mark)[:5].tolist()}')

    # 检查缺失类别
    if len(uniq) > 0:
        full = np.arange(uniq.min(), uniq.max() + 1)
        missing = np.setdiff1d(full, uniq)
        print(f'label 值域 [{uniq.min()}, {uniq.max()}] 内缺失的类别数: {len(missing)}')
        if len(missing) > 0 and len(missing) <= 30:
            print(f'缺失的类别: {missing.tolist()}')

    # bincount 对照
    counts = np.bincount(label.astype(np.int64))
    print(f'bincount 长度={len(counts)}, 非零桶数={np.count_nonzero(counts)}, '
          f'非零桶平均样本数={counts[counts > 0].mean():.2f}')
    print()
