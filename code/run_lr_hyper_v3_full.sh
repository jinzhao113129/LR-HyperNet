#!/bin/bash
#SBATCH -J lr_hyper_v3_full
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_full_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_full_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v3 FULL (KDA + global_anchor) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# 与 SECNet 复现 / v1 / v2 完全同协议: AdamW + 150ep + batch80 + 10240点 + official_gen
# v3 增强: KDA 因果时间编码(--kda, Kimi Delta Attention 式 gated delta rule) + 全局吸收节点(--use_global_anchor)
# 注意: --kda 与 --time_enc 互斥（lr_hypernet.py 内 raise），不要同时传
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 80 \
    --epoch 150 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_gen/ \
    --log_dir lr_hyper_v3_full_150ep_adamw_a800 \
    --log_name /lr_hyper_v3_full \
    --remote_g 16 \
    --curvature_init 0.4 \
    --kda \
    --use_global_anchor
echo "=== LR-HyperNet v3 FULL end: $(date) ==="
