#!/bin/bash
#SBATCH -J infer_rp1_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o eval_results/infer_rp1_ai2_%j.out
#SBATCH -e eval_results/infer_rp1_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_rp1_fix_v100 \
    --which both \
    --eval_seed 42 \
    --batch_size 8 \
    --gpu 0