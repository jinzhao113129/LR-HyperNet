#!/bin/bash
# 测试 Claude Code 是否可用
cd ~/JiangYouxin/ZhaoJin

echo "=== 测试 Claude Code ==="
echo "路径: /home/jnguan/zhangchenyuan/node/current/bin/claude"
echo ""

# 测试1: 版本号
echo "--- 测试1: 版本号 ---"
/home/jnguan/zhangchenyuan/node/current/bin/claude --version 2>&1
echo ""

# 测试2: 简单非交互任务（限时60秒）
echo "--- 测试2: 简单任务（写一个hello.py）---"
timeout 60 /home/jnguan/zhangchenyuan/node/current/bin/claude -p '请在当前目录创建一个文件 hello.py，内容是 print("Hello from Claude Code")，然后用 python 运行它。只做这件事。' 2>&1
echo ""

# 测试3: 检查文件是否生成
echo "--- 测试3: 检查文件 ---"
if [ -f hello.py ]; then
    echo "✅ hello.py 存在，内容:"
    cat hello.py
    echo ""
    echo "运行结果:"
    ~/JiangYouxin/envs/eventprobe/bin/python hello.py
else
    echo "❌ hello.py 不存在，Claude 可能没有完成任务"
fi

echo ""
echo "=== 测试完成 ==="
