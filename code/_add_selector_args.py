"""添加 selector_mode 和 div_lambda 参数透传到 lr_hypernet.py 和 train_lr_hyper.py"""
import re

# === 修改 lr_hypernet.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 __init__ 参数
content = content.replace(
    "                 use_rp1_selector: bool = False,\n                 rp1_lambda: float = 1.0):",
    "                 use_rp1_selector: bool = False,\n                 rp1_lambda: float = 1.0,\n                 selector_mode: str = \"topk\",\n                 div_lambda: float = 1.0):"
)

# 2. 添加 docstring 说明（在 rp1_lambda 说明后）
content = content.replace(
    "            rp1_lambda: 空间新颖性权重（仅 use_rp1_selector=True 时有效）",
    "            rp1_lambda: 空间新颖性权重（仅 use_rp1_selector=True 时有效）\n            selector_mode: 远程点选择模式，\"topk\"（独立打分，默认）或 \"greedy_div\"（贪心最大-最小多样性，RP1启发）\n            div_lambda: 多样性权重（仅 selector_mode=\"greedy_div\" 时有效），越大远程点越分散"
)

# 3. LocalRemoteGrouper 初始化透传
content = content.replace(
    "                use_rp1_selector=use_rp1_selector,\n                rp1_lambda=rp1_lambda)",
    "                use_rp1_selector=use_rp1_selector,\n                rp1_lambda=rp1_lambda,\n                selector_mode=selector_mode,\n                div_lambda=div_lambda)"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\lr_hypernet.py", "w", encoding="utf-8") as f:
    f.write(content)

print("lr_hypernet.py 修改完成")

# === 修改 train_lr_hyper.py ===
with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 添加 argparse 参数（在 rp1_lambda 后）
content = content.replace(
    "    parser.add_argument('--rp1_lambda', type=float, default=1.0, help='RP1空间新颖性权重')",
    "    parser.add_argument('--rp1_lambda', type=float, default=1.0, help='RP1空间新颖性权重')\n    parser.add_argument('--selector_mode', type=str, default='topk', choices=['topk', 'greedy_div'], help='远程点选择模式: topk独立打分 / greedy_div贪心多样性')\n    parser.add_argument('--div_lambda', type=float, default=1.0, help='贪心多样性权重(仅greedy_div)')"
)

# 2. 模型构造透传（在 rp1_lambda 后）
content = content.replace(
    "        use_rp1_selector=args.use_rp1_selector,\n        rp1_lambda=args.rp1_lambda",
    "        use_rp1_selector=args.use_rp1_selector,\n        rp1_lambda=args.rp1_lambda,\n        selector_mode=args.selector_mode,\n        div_lambda=args.div_lambda"
)

with open(r"D:\UserData\Desktop\学习\进组\secnet-hyperbolic\code\train_lr_hyper.py", "w", encoding="utf-8") as f:
    f.write(content)

print("train_lr_hyper.py 修改完成")
