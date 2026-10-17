#!/bin/bash
#SBATCH -J inf_s7base_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o eval_results/infer_s7base_ai2_%j.out
#SBATCH -e eval_results/infer_s7base_ai2_%j.err
# 2026-10-14 16:40 ai2: seed7 基线（173249）训练完成后的固定 eval_seed=42 推理
# last + best 双口径，产出 pred_seed42.npz 供 McNemar 对比（基线 seed42 = 46.1456%）
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_baseline_s7_ai2 \
    --which last \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_baseline_s7_ai2 \
    --which best \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
