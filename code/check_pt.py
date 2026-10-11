import torch
import sys

path = "/home/jnguan/JiangYouxin/data/event_datasets/ncaltech101_h5_secnet_np8192_w30ms/test.pt"
d = torch.load(path, map_location="cpu", weights_only=False)
print("type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys()))
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}, value={v}")
elif isinstance(d, (list, tuple)):
    print("len:", len(d))
    for i, v in enumerate(d):
        if hasattr(v, "shape"):
            print(f"  [{i}]: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  [{i}]: type={type(v)}")
else:
    print("unknown type")
