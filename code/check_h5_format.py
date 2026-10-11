"""快速检查 h5 数据格式"""
import h5py
import numpy as np

with h5py.File('/home/jnguan/JiangYouxin/ZhaoJin/data/train.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    label = f['label'][:]
    mark = f['mark'][:]
    data = f['data'][:]

print("label shape:", label.shape)
print("label dtype:", label.dtype)
print("label unique count:", len(np.unique(label)))
print("label min:", label.min(), "max:", label.max())
print("label unique (first 30):", np.sort(np.unique(label))[:30])
print("label unique (last 20):", np.sort(np.unique(label))[-20:])
print()
print("mark shape:", mark.shape)
print("mark min:", mark.min(), "max:", mark.max())
print("mark unique count:", len(np.unique(mark)))
print()
print("data shape:", data.shape)
print("data dtype:", data.dtype)
print("data[0, 0, :]:", data[0, 0, :])
print("data[0, :, 0] (x) range:", data[:, :, 0].min(), data[:, :, 0].max())
print("data[0, :, 1] (y) range:", data[:, :, 1].min(), data[:, :, 1].max())
print("data[0, :, 2] (t) range:", data[:, :, 2].min(), data[:, :, 2].max())
print("data[0, :, 3] (p) unique:", np.unique(data[:, :, 3]))
