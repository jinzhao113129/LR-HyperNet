#!/bin/bash
#SBATCH -J lr_hyper_v2_anchor
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v2_anchor_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v2_anchor_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v2 ANCHOR-only (use_global_anchor) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# 消融2: 只有全局吸收节点, 无时间编码
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 80 \
    --epoch 150 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_gen/ \
    --log_dir lr_hyper_v2_anchor_150ep_adamw_a800 \
    --remote_g 16 \
    --curvature_init 0.4 \
    --use_global_anchor
echo "=== LR-HyperNet v2 ANCHOR-only end: $(date) ==="
