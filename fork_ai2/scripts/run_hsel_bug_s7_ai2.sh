#!/bin/bash
#SBATCH -J sn_s7_hsel_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_s7_hsel_ai2_%j.out
#SBATCH -e eval_results/sn_s7_hsel_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: P1-A1归因修正核心证据的跨种子复核。
# hsel_bug 配置（双曲网格选点+SECNet原生缩放坐标，seed42 时 +4.71pp p=0.0002）
# 换训练 seed=7 复核。2x2 矩阵"缩放坐标+双曲网格"象限。
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 60 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_hsel_bug_s7_ai2 \
    --log_name /secnet_hsel_bug_s7_ai2 \
    --grad_accum 4 \
    --seed 7 \
    --hyperbolic_selector
