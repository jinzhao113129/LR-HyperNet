#!/bin/bash
#SBATCH -J lr_hyper_v3_smoke
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_smoke_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_smoke_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v3 SMOKE (KDA + global_anchor) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# 1) 显存实测：batch=80 前向+反向峰值（Claude 审查建议项）
/home/jnguan/.miniconda/envs/bhli/bin/python _v3_mem_check.py || exit 1

# 2) 1 epoch 冒烟：同协议跑通一个 epoch（log_dir/log_name 独立，不污染正式实验）
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 80 \
    --epoch 1 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_gen/ \
    --log_dir lr_hyper_v3_smoke \
    --log_name /lr_hyper_v3_smoke \
    --remote_g 16 \
    --curvature_init 0.4 \
    --kda \
    --use_global_anchor
echo "=== LR-HyperNet v3 SMOKE end: $(date) ==="
