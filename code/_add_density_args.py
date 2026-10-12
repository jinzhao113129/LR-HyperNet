"""添加 density_channel 和 density_bins 参数透传到 train_lr_hyper.py"""

# === 修改 train_lr_hyper.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 argparse 参数
content = content.replace(
    "    parser.add_argument('--remote_kda', action='store_true', help='EventMamba启发的远程KDA扫描')",
    "    parser.add_argument('--remote_kda', action='store_true', help='EventMamba启发的远程KDA扫描')\n    parser.add_argument('--density_channel', action='store_true', help='SpikePoint启发的密度输入通道(32x32体素log压缩)')\n    parser.add_argument('--density_bins', type=int, default=32, help='密度体素bin数(默认32)')"
)

# 2. 模型构造透传
content = content.replace(
    "        remote_kda=args.remote_kda",
    "        remote_kda=args.remote_kda,\n        density_channel=args.density_channel,\n        density_bins=args.density_bins"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "w", encoding="utf-8") as f:
    f.write(content)

print("train_lr_hyper.py 修改完成")

# 验证 lr_hypernet.py 中已有 density_channel 参数
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "r", encoding="utf-8") as f:
    lr_content = f.read()

assert "density_channel" in lr_content, "lr_hypernet.py 缺少 density_channel 参数"
assert "density_bins" in lr_content, "lr_hypernet.py 缺少 density_bins 参数"
print("lr_hypernet.py 已包含 density_channel/density_bins 参数 ✓")
