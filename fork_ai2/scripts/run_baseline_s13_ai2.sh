#!/bin/bash
#SBATCH -J sn_s13_base_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_s13_base_ai2_%j.out
#SBATCH -e eval_results/sn_s13_base_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: P1 跨种子复核第三组——基线 seed13。
# 目的：ha_rp1 组合增益的跨种子稳健性验证（seed42/seed7 之外再补一个种子）。
# 同一配置仅换 seed，其余与 secnet_baseline_s7_ai2 完全一致。
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 60 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_baseline_s13_ai2 \
    --log_name /secnet_baseline_s13_ai2 \
    --grad_accum 4 \
    --seed 13
