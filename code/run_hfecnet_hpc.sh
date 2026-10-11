#!/bin/bash
#SBATCH -J hfecnet_large
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/hfecnet_large_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/hfecnet_large_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== HFECNet Mobius large config (official scale) start: $(date) ==="
echo "Host: $(hostname)"
# HFECNet 修正版：feature_list [64,132,268,540], group_number [2048,1024,512,256]
# + Möbius 线性变换（切空间 W·log_o(x)+b → exp 回双曲，参考 ECAF-GCN）
# 与 SECNet 官方同规模，公平对比。batch 32 卡在 stage3 特征+双曲映射，降为 16。
/home/jnguan/.miniconda/envs/bhli/bin/python train_hfecnet.py \
    --batch_size 16 \
    --epoch 150 \
    --num_category 101 \
    --num_point 8192 \
    --data_path ./data/ \
    --log_dir hfecnet_mobius_150 \
    --ablation none
echo "=== HFECNet Mobius large config end: $(date) ==="
