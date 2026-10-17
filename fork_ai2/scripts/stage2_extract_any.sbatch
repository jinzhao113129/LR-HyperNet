#!/bin/bash
#SBATCH -J s2_feat
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/stage2/logs/s2_feat_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/stage2/logs/s2_feat_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

cd /home/jnguan/JiangYouxin/ZhaoJin/stage2
mkdir -p logs features
echo "=== stage2 feature extraction start: $(date) ==="
nvidia-smi | head -12
/home/jnguan/.miniconda/envs/bhli/bin/python stage2_extract_features.py \
    --data_path /home/jnguan/JiangYouxin/ZhaoJin/ncaltech101_official_gen/ \
    --ckpt /home/jnguan/JiangYouxin/ZhaoJin/archive/old_checkpoints/checkpoint_official/best_checkpoint.pth \
    --split both --batch 32
echo "=== extraction done: $(date) ==="

/home/jnguan/.miniconda/envs/bhli/bin/python stage2_probe_official.py
echo "=== probe done: $(date) ==="
