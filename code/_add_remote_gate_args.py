"""添加 remote_gate 参数透传到 lr_hypernet.py 和 train_lr_hyper.py"""

# === 修改 lr_hypernet.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 __init__ 参数
content = content.replace(
    "                 selector_mode: str = \"topk\",\n                 div_lambda: float = 1.0):",
    "                 selector_mode: str = \"topk\",\n                 div_lambda: float = 1.0,\n                 remote_gate: bool = False):"
)

# 2. 添加 docstring
content = content.replace(
    "            div_lambda: 多样性权重（仅 selector_mode=\"greedy_div\" 时有效），越大远程点越分散",
    "            div_lambda: 多样性权重（仅 selector_mode=\"greedy_div\" 时有效），越大远程点越分散\n            remote_gate: SECNet式远程分支门控融合。中心点特征生成sigmoid门控，乘性调制远程特征（非加性），初始化bias=2.0起步等价直连"
)

# 3. LocalRemoteGrouper 初始化透传
content = content.replace(
    "                selector_mode=selector_mode,\n                div_lambda=div_lambda)",
    "                selector_mode=selector_mode,\n                div_lambda=div_lambda,\n                remote_gate=remote_gate)"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "w", encoding="utf-8") as f:
    f.write(content)

print("lr_hypernet.py 修改完成")

# === 修改 train_lr_hyper.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 argparse 参数
content = content.replace(
    "    parser.add_argument('--div_lambda', type=float, default=1.0, help='贪心多样性权重(仅greedy_div)')",
    "    parser.add_argument('--div_lambda', type=float, default=1.0, help='贪心多样性权重(仅greedy_div)')\n    parser.add_argument('--remote_gate', action='store_true', help='SECNet式远程分支门控融合')"
)

# 2. 模型构造透传
content = content.replace(
    "        selector_mode=args.selector_mode,\n        div_lambda=args.div_lambda",
    "        selector_mode=args.selector_mode,\n        div_lambda=args.div_lambda,\n        remote_gate=args.remote_gate"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "w", encoding="utf-8") as f:
    f.write(content)

print("train_lr_hyper.py 修改完成")
