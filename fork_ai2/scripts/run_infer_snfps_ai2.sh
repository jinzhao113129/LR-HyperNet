#!/bin/bash
#SBATCH -J infer_snfps_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o eval_results/infer_snfps_ai2_%j.out
#SBATCH -e eval_results/infer_snfps_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: extract_preds.py 已补齐 scale_norm_fps/hyperbolic_mlp 两个 flag，
# 旧结果 last_seed42.npz acc=2.570% 是模型按 flag=False 重建导致的无效值，重推。
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_scale_norm_fps_v100 \
    --which both \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0
