#!/bin/bash
#SBATCH -J lr_hyper_v3_v100sub50
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_v100sub50_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_v100sub50_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 6-00:00:00
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v3 V100 subset (KDA + global_anchor) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# V100-16GB batch=16 快速版（30ep + 25% 子集，与 165999/166000 同协议，sanity/相对趋势用，不进论文）
# 2026-10-10 改跑 H100（gpu04 空卡）：V100-32G 排队一天进不去
# v3 增强：KDA 因果时间编码(--kda, Kimi Delta Attention 式 gated delta rule) + 全局吸收节点(--use_global_anchor)
# 注意: --kda 与 --time_enc 互斥（lr_hypernet.py 内 raise），不要同时传
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 16 \
    --epoch 30 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_sub/ \
    --log_dir lr_hyper_v3_full_sub50_v100b16 \
    --log_name /lr_hyper_v3_v100sub50 \
    --remote_g 16 \
    --curvature_init 0.4 \
    --kda \
    --use_global_anchor
echo "=== LR-HyperNet v3 V100 subset end: $(date) ==="
