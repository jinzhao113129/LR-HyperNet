#!/bin/bash
#SBATCH -J infer_best_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o eval_results/infer_best_ai2_%j.out
#SBATCH -e eval_results/infer_best_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: extract_preds.py 已改为每个 (log_dir, which) 独立重播种，
# best 值不再随调用方式漂移。重推三个有 best checkpoint 的 P0 实验（仅 best）。
# RP1 的旧 best_seed42.npz（49.251）是非标准调用产生的污染文件，本次覆盖修正。
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_rp1_fix_v100 \
    --which best \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_kda_v100 \
    --which best \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_scale_norm_fps_v100 \
    --which best \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_hyperattn_rp1_fix_v100 \
    --which best \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
