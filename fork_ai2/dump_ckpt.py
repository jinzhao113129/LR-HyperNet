import torch
c = torch.load("archive/old_checkpoints/checkpoint_official/best_checkpoint.pth", map_location="cpu")
net = c.get("net", c)
ks = list(net.keys())
print("TOTAL_KEYS", len(ks))
print("TOP_GROUPS", sorted(set(k.split(".")[0] for k in ks)))
print("BN_LOCAL", [k for k in ks if "bn_local" in k])
print("BN_GLOBAL", [k for k in ks if "bn_global" in k])
print("GROUP_CONV", [k for k in ks if "group_conv" in k])
print("CONV_LIST", [k for k in ks if "conv_list" in k])
cls = [(k, tuple(v.shape)) for k, v in net.items() if "classifier" in k]
print("CLASSIFIER", cls[:12])
print("best_test_acc", c.get("best_test_acc"))
print("acc", c.get("acc"))