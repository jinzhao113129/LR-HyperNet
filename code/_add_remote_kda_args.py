"""添加 remote_kda 参数透传到 lr_hypernet.py 和 train_lr_hyper.py"""

# === 修改 lr_hypernet.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 __init__ 参数
content = content.replace(
    "                 remote_gate: bool = False):",
    "                 remote_gate: bool = False,\n                 remote_kda: bool = False):"
)

# 2. 添加 docstring
content = content.replace(
    "            remote_gate: SECNet式远程分支门控融合。中心点特征生成sigmoid门控，乘性调制远程特征（非加性），初始化bias=2.0起步等价直连",
    "            remote_gate: SECNet式远程分支门控融合。中心点特征生成sigmoid门控，乘性调制远程特征（非加性），初始化bias=2.0起步等价直连\n            remote_kda: EventMamba启发的远程KDA扫描。复用KDACausalEncoder对远程16点按时间排序做因果递推，捕获长程时序演变，残差加法"
)

# 3. LocalRemoteGrouper 初始化透传
content = content.replace(
    "                remote_gate=remote_gate)",
    "                remote_gate=remote_gate,\n                remote_kda=remote_kda)"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "w", encoding="utf-8") as f:
    f.write(content)

print("lr_hypernet.py 修改完成")

# === 修改 train_lr_hyper.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 argparse 参数
content = content.replace(
    "    parser.add_argument('--remote_gate', action='store_true', help='SECNet式远程分支门控融合')",
    "    parser.add_argument('--remote_gate', action='store_true', help='SECNet式远程分支门控融合')\n    parser.add_argument('--remote_kda', action='store_true', help='EventMamba启发的远程KDA扫描')"
)

# 2. 模型构造透传
content = content.replace(
    "        remote_gate=args.remote_gate",
    "        remote_gate=args.remote_gate,\n        remote_kda=args.remote_kda"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "w", encoding="utf-8") as f:
    f.write(content)

print("train_lr_hyper.py 修改完成")
