#!/bin/bash
# 分桶分析 v2：GPU 版（CPU 版太慢）
# 对比 v3.1双曲 vs 欧氏对照 vs 基线b8，按训练集每类样本数分桶
#SBATCH -J bucket_analysis_v2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/bucket_analysis_v2_%j.out
#SBATCH -e eval_results/bucket_analysis_v2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== Bucket Analysis v2 start: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python bucket_analysis_v2.py

echo "=== done: $(date) ==="
