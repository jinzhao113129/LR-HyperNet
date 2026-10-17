#!/bin/bash
#SBATCH -J sn_frz_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_frz_ai2_%j.out
#SBATCH -e eval_results/sn_frz_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: A1归因修正对照——SECNet 原生可学习缩放冻结为 1.0。
# 量化 SECNet 隐藏组件（fps_parameter 逐维缩放）对基线的独立贡献。
# 依赖 train_frequency_ncaltech.py 新增的 --freeze_fps_parameter flag。
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 60 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_frz_fps_ai2 \
    --log_name /secnet_frz_fps_ai2 \
    --grad_accum 4 \
    --seed 42 \
    --freeze_fps_parameter
