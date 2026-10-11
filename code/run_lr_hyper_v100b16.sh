#!/bin/bash
#SBATCH -J lr_hyper_v100b16
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v100b16_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v100b16_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 6-00:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v1 V100 batch16 start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# V100-16GB batch=16 快速版（从零，独立 log_dir，不续 A100 的 checkpoint）
# 协议除 batch 外与正式版一致：AdamW + 150ep + 10240点 + official_gen
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 16 \
    --epoch 150 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_gen/ \
    --log_dir lr_hyper_official_gen_150ep_adamw_v100b16 \
    --log_name /lr_hyper_v100b16 \
    --remote_g 16 \
    --curvature_init 0.4
echo "=== LR-HyperNet v1 V100 batch16 end: $(date) ==="
