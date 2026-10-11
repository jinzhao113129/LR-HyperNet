import torch
import numpy as np

path = "/home/jnguan/JiangYouxin/data/event_datasets/ncaltech101_h5_secnet_np8192_w30ms/test.pt"
d = torch.load(path, map_location="cpu", weights_only=False)

print("=== data shape ===")
print(d["data"].shape, d["data"].dtype)
print("=== label ===")
print(d["label"].shape, d["label"].dtype)
print("label unique count:", len(torch.unique(d["label"])))
print("label min/max:", d["label"].min().item(), d["label"].max().item())
print("=== mark ===")
print(d["mark"].shape, d["mark"].dtype)
mark_np = d["mark"].numpy() if hasattr(d["mark"], "numpy") else np.array(d["mark"])
print("mark unique values:", np.unique(mark_np))
print("mark value counts:")
for v in np.unique(mark_np):
    print(f"  {v}: {np.sum(mark_np == v)}")
print("=== sample_name unique ===")
names = d["sample_name"]
print("total names:", len(names))
print("unique names:", len(set(names)))
print("first 3 unique:", list(set(names))[:3])
