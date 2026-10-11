"""
将 SECNet 格式的 .pt 数据转换为 .h5 格式，并按 mark（原始图片索引）8:2 分割 train/test。
输入: /home/jnguan/JiangYouxin/data/event_datasets/ncaltech101_h5_secnet_np8192_w30ms/test.pt
输出: ~/JiangYouxin/ZhaoJin/data/train.h5, test.h5
"""
import torch
import h5py
import numpy as np
import os

INPUT_PATH = "/home/jnguan/JiangYouxin/data/event_datasets/ncaltech101_h5_secnet_np8192_w30ms/test.pt"
OUTPUT_DIR = os.path.expanduser("~/JiangYouxin/ZhaoJin/data")
SEED = 42
TRAIN_RATIO = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {INPUT_PATH} ...")
d = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
data = d["data"].numpy() if hasattr(d["data"], "numpy") else np.array(d["data"])  # [N, 8192, 4]
label = d["label"].numpy() if hasattr(d["label"], "numpy") else np.array(d["label"])  # [N]
mark = np.array(d["mark"])  # [N]

print(f"Total: {data.shape[0]} samples, {data.shape[1]} points, {len(np.unique(label))} classes")
print(f"Unique marks (images): {len(np.unique(mark))}")

# 按 mark 分割
rng = np.random.RandomState(SEED)
unique_marks = np.unique(mark)
rng.shuffle(unique_marks)
n_train = int(len(unique_marks) * TRAIN_RATIO)
train_marks = set(unique_marks[:n_train].tolist())
test_marks = set(unique_marks[n_train:].tolist())

train_mask = np.array([m in train_marks for m in mark])
test_mask = ~train_mask

train_data = data[train_mask]
train_label = label[train_mask]
train_mark = mark[train_mask]
test_data = data[test_mask]
test_label = label[test_mask]
test_mark = mark[test_mask]

print(f"Train: {train_data.shape[0]} samples ({len(train_marks)} images)")
print(f"Test:  {test_data.shape[0]} samples ({len(test_marks)} images)")

# 保存为 h5
for split, d, l, m in [("train", train_data, train_label, train_mark),
                         ("test", test_data, test_label, test_mark)]:
    out_path = os.path.join(OUTPUT_DIR, f"{split}.h5")
    print(f"Writing {out_path} ...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=d.astype(np.float32), compression="gzip")
        f.create_dataset("label", data=l.astype(np.int64), compression="gzip")
        f.create_dataset("mark", data=m.astype(np.int64), compression="gzip")
    print(f"  data: {d.shape}, label: {l.shape}, mark: {m.shape}")

print("Done!")
