#!/bin/bash
#SBATCH -J lr_hyper_v3_smoke_h100
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_smoke_h100_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_smoke_h100_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v3 SMOKE-V100 (KDA 修复版: beta-5/alpha-lower/scale0.1/lr1-10) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 子集 2 epoch 冒烟：确认 NaN 修复 + 能学习（log_dir 独立）
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 8 \
    --epoch 2 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_sub/ \
    --log_dir lr_hyper_v3_smoke_h100 \
    --log_name /lr_hyper_v3_smoke_h100 \
    --remote_g 16 \
    --curvature_init 0.4 \
    --kda \
    --use_global_anchor
echo "=== LR-HyperNet v3 SMOKE-H100 end: $(date) ==="
