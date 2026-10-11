#!/bin/bash
#SBATCH -J lr_hyper_v2_time
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v2_time_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v2_time_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v2 TIME-only (time_enc) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# 消融1: 只有时间因果序编码, 无全局锚点
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 80 \
    --epoch 150 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_gen/ \
    --log_dir lr_hyper_v2_time_150ep_adamw_a800 \
    --remote_g 16 \
    --curvature_init 0.4 \
    --time_enc
echo "=== LR-HyperNet v2 TIME-only end: $(date) ==="
