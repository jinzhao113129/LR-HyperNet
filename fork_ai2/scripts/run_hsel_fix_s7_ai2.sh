#!/bin/bash
#SBATCH -J sn_s7_hselfix_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_s7_hselfix_ai2_%j.out
#SBATCH -e eval_results/sn_s7_hselfix_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: 2×2矩阵"双曲网格+原始坐标"象限（修复版）的seed7复核。
# 显式 flag：--hyperbolic_selector（选点用 fps_parameter 缩放前原始 x,y，
# 即修复版行为）。seed42 时 48.18% +2.03pp p=0.107 不显著。
# （取代被取消的 173350：该作业实际跑的就是这个配置，但脚本名误标
# hsel_bug。现更名重提，配置显式。）
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 60 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_hsel_fix_s7_ai2 \
    --log_name /secnet_hsel_fix_s7_ai2 \
    --grad_accum 4 \
    --seed 7 \
    --hyperbolic_selector
