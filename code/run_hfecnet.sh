#!/bin/bash
# HFECNet 双曲模型训练启动脚本
# 用法: bash run_hfecnet.sh
# 输出: logs/hfecnet_train.log

set -e

cd ~/JiangYouxin/ZhaoJin
mkdir -p logs checkpoints

PYTHON=~/JiangYouxin/envs/eventprobe/bin/python

echo "=== HFECNet 双曲模型训练 ==="
echo "开始时间: $(date)"
echo "数据: data/train.h5 (14297样本), data/test.h5 (3572样本)"
echo "配置: num_point=8192, num_category=101, batch_size=64, epoch=150"
echo "双曲曲率初始值: 0.4"
echo ""

nohup $PYTHON train_hfecnet.py \
    --batch_size 64 \
    --epoch 150 \
    --num_point 8192 \
    --num_category 101 \
    --data_path ./data/ \
    --curvature_init 0.4 \
    --log_dir hfecnet_hyperbolic \
    > logs/hfecnet_train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f logs/hfecnet_train.log"
echo "查看进程: ps aux | grep train_hfecnet"
