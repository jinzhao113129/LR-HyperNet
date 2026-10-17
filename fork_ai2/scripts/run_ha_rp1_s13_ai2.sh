#!/bin/bash
#SBATCH -J sn_s13_harp_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_s13_harp_ai2_%j.out
#SBATCH -e eval_results/sn_s13_harp_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: P1 跨种子复核第三组——ha_rp1（双曲注意力+RP1网格选点）seed13。
# 目的：验证组合增益跨 3 个种子（42/7/13）的稳健性。seed42 时 ha_rp1 50.107%
# +3.96pp p=0.0023 为最强组件。配置与 secnet_ha_rp1_s7_ai2 一致，仅换 seed。
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 60 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_ha_rp1_s13_ai2 \
    --log_name /secnet_ha_rp1_s13_ai2 \
    --grad_accum 4 \
    --seed 13 \
    --hyperbolic_attention --rp1_selector
