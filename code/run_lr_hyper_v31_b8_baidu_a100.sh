#!/bin/bash
# LR-HyperNet v3.1 小版本测试（百度网盘官方数据）
# v3.1 = v1 + 曲率优化三件套 + Lorentz 模型 + time_enc（修复版）
# 放弃 KDA（24步递推梯度爆炸，两次 epoch8 崩溃）
#SBATCH -J lr_hyper_v31_b8_baidu
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH -t 10:00:00
#SBATCH --exclude=gpu02,gpu15
#SBATCH -o log/slurm-%j.out
#SBATCH -e log/slurm-%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PATH=/home/jnguan/.miniconda/envs/bhli/bin:$PATH

python train_lr_hyper.py \
    --batch_size 8 \
    --epoch 30 \
    --num_point 10240 \
    --data_path ./ncaltech101_official_baidu/ \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 1e-4 \
    --curvature_init 0.4 \
    --remote_g 16 \
    --time_enc \
    --hyperbolic_type lorentz \
    --log_dir lr_hyper_v31_b8_baidu_a100 \
    --log_name /lr_hyper_v31_b8_baidu
