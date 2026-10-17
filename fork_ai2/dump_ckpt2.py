import torch
c = torch.load("archive/old_checkpoints/checkpoint_official/best_checkpoint.pth", map_location="cpu")
net = c.get("net", c)
def stat(t, name):
    t = t.float()
    print(name, "shape", tuple(t.shape), "mean", round(t.mean().item(),5), "std", round(t.std().item(),5), "min", round(t.min().item(),5), "max", round(t.max().item(),5))
stat(net["model.bn_local_list.0.weight"], "bn_local0.weight")
stat(net["model.bn_local_list.0.running_var"], "bn_local0.running_var")
stat(net["model.bn_local_list.0.running_mean"], "bn_local0.running_mean")
stat(net["model.bn_global_list.0.weight"], "bn_global0.weight")
stat(net["model.bn_global_list.0.running_var"], "bn_global0.running_var")
stat(net["model.group_conv_list.0.net1.0.weight"], "group_conv0.net1.0.weight")
stat(net["model.conv_list.0.net1.0.weight"], "conv0.net1.0.weight")
stat(net["model.embed_dim.net.0.weight"], "embed_dim.net.0.weight")