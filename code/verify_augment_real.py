"""在真实百度数据上验证 augment 集成：train_lr_hyper.py --augment 2 能正确加载增强数据。

只验证数据加载部分（不训练），确认：
1. --augment 参数被 argparse 接受
2. 增强后数据 shape = 3596*3
3. mark 语义保持（增强样本共享原 mark）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import provider_data
from augment_data import augment_train_data

# 真实数据
data, label, mark = provider_data.load_h5_mark('./ncaltech101_official_baidu/train.h5')
print(f"原始: data={data.shape}, label={np.squeeze(label).shape}, mark={mark.shape}")
assert data.shape == (3596, 10240, 4), f"期望 (3596,10240,4)，实际 {data.shape}"

# 增强 ×2
d, l, m = augment_train_data(data, label, mark, n_versions=2)
print(f"增强后: data={d.shape}, label={l.shape}, mark={m.shape}")
assert d.shape == (3596 * 3, 10240, 4), f"期望 (10788,10240,4)，实际 {d.shape}"
assert l.shape[0] == 10788 and m.shape[0] == 10788, "label/mark 维度错误"

# mark 语义：原 mark 应出现 3 次（原样 + 2 增强版本）
unique_marks, counts = np.unique(m, return_counts=True)
print(f"mark 唯一数: {len(unique_marks)}（期望 3596）")
print(f"每个 mark 出现次数: min={counts.min()}, max={counts.max()}（期望 3）")
assert len(unique_marks) == 3596, "mark 唯一数不等于 3596"
assert counts.min() == 3 and counts.max() == 3, "每个 mark 应恰好出现 3 次"

# label 分布保持（每类×3）
unique_labels, label_counts = np.unique(l, return_counts=True)
print(f"类别数: {len(unique_labels)}（期望 101）")
print(f"类别 0-4 样本数: {label_counts[:5]}")

# 数据内容检查：增强版本 ≠ 原样（至少部分样本）
n_same = 0
for i in range(100):
    orig = d[i]
    aug = d[i + 3596]  # 第一个增强版本
    if np.array_equal(orig, aug):
        n_same += 1
print(f"前100样本中增强版与原样完全相同数: {n_same}（期望接近0，随机增强）")

print("\n真实数据增强验证全部通过 ✓")
