#!/usr/bin/env python3
"""从 ncaltech101_official_gen 采样 25% 训练子集（按视频 mark 分层），test 保持全量。
输出 ncaltech101_sub/{train.h5, test.h5}
"""
import h5py
import numpy as np
import os

SRC = "ncaltech101_official_gen"
DST = "ncaltech101_sub"
RATIO = 0.25
SEED = 2026

os.makedirs(DST, exist_ok=True)

with h5py.File(f"{SRC}/train.h5", "r") as f:
    data = f["data"][:]
    label = f["label"][:]
    mark = f["mark"][:]

print("train full:", data.shape, data.dtype)
print("labels:", np.unique(label).size, "marks:", np.unique(mark).size)

# 按 mark 分组：同视频所有窗口共享 mark
marks, inv = np.unique(mark, return_inverse=True)
nmarks = marks.size
mark_label = np.zeros(nmarks, dtype=np.int64)
for i, m in enumerate(marks):
    mark_label[i] = label[inv == i][0]

# 按类分层，每类取 25% 的 mark（至少 1 个）
rng = np.random.default_rng(SEED)
sel = np.zeros(nmarks, dtype=bool)
for c in np.unique(mark_label):
    cand = np.where(mark_label == c)[0]
    k = max(1, int(np.ceil(RATIO * len(cand))))
    sel[rng.choice(cand, size=k, replace=False)] = True

keep_marks = marks[sel]
keep_idx = np.isin(mark, keep_marks)
print("sampled marks:", keep_marks.size, "/", nmarks, "-> train samples:", int(keep_idx.sum()))

with h5py.File(f"{DST}/train.h5", "w") as f:
    f.create_dataset("data", data=data[keep_idx])
    f.create_dataset("label", data=label[keep_idx])
    f.create_dataset("mark", data=mark[keep_idx])

# test 全量拷贝
os.system(f"cp {SRC}/test.h5 {DST}/test.h5")

with h5py.File(f"{DST}/test.h5", "r") as f:
    print("test:", f["data"].shape, "keys:", list(f.keys()))

# 校验：子集内 label 与 mark 分组一致性
with h5py.File(f"{DST}/train.h5", "r") as f:
    l2 = f["label"][:]
    m2 = f["mark"][:]
    ok = True
    for m in np.unique(m2):
        ls = l2[m2 == m]
        if np.unique(ls).size != 1:
            ok = False
            break
    print("mark-group label consistency:", ok)
print("done. DST:", DST)
