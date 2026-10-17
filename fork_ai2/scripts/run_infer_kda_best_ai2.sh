#!/bin/bash
#SBATCH -J infer_kda_best_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o eval_results/infer_kda_best_ai2_%j.out
#SBATCH -e eval_results/infer_kda_best_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: KDA 修复版只推过 last，补 best 以便与 RP1（last+best 都有）对齐。
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_kda_v100 \
    --which best \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
