#!/bin/bash
#SBATCH -J lr_hyper_v2_full
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v2_full_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v2_full_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v2 FULL (time_enc + global_anchor) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# 与 SECNet 复现 / v1 完全同协议: AdamW + 150ep + batch80 + 10240点 + official_gen
# v2 增强: 时间因果序编码(--time_enc) + 全局吸收节点(--use_global_anchor)
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 80 \
    --epoch 150 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_gen/ \
    --log_dir lr_hyper_v2_full_150ep_adamw_a800 \
    --log_name /lr_hyper_v2_full \
    --remote_g 16 \
    --curvature_init 0.4 \
    --time_enc \
    --use_global_anchor
echo "=== LR-HyperNet v2 FULL end: $(date) ==="
