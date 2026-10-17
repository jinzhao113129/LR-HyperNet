#!/bin/bash
#SBATCH -J sn_s7_hselbug_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_s7_hselbug_ai2_%j.out
#SBATCH -e eval_results/sn_s7_hselbug_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: 2×2矩阵"双曲网格+缩放坐标"象限（旧版bug行为）的seed7复核。
# 显式 flag 组合：--hyperbolic_selector --hsel_scaled_coords
# （前者=双曲网格选点，后者=选点用 fps_parameter 缩放后坐标，复现旧版bug路径）
# seed42 时该配置 +4.71pp (p=0.0002) 为最强组件，需跨种子验证。
# 与被取消的旧 173350 的区别：旧作业只传了 --hyperbolic_selector（修复版
# 行为）却命名为 hsel_bug——现已显式化，不再靠脚本名区分。
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
    --hyperbolic_selector \
    --hsel_scaled_coords
