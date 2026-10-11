#!/bin/bash
# SECNet 基线训练启动脚本
# 用法: bash run_secnet.sh
# 输出: logs/secnet_train.log

set -e

cd ~/JiangYouxin/ZhaoJin
mkdir -p logs checkpoints

PYTHON=~/JiangYouxin/envs/eventprobe/bin/python

echo "=== SECNet 基线训练 ==="
echo "开始时间: $(date)"
echo "数据: data/train.h5 (14297样本), data/test.h5 (3572样本)"
echo "配置: num_point=8192, num_category=101, batch_size=64, epoch=150"
echo ""

nohup $PYTHON train_frequency.py \
    --batch_size 64 \
    --epoch 150 \
    --num_point 8192 \
    --num_category 101 \
    --data_path ./data/ \
    --log_dir secnet_baseline \
    > logs/secnet_train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f logs/secnet_train.log"
echo "查看进程: ps aux | grep train_frequency"
