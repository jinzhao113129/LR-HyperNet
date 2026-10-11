#!/bin/bash
#SBATCH -J vote_ncaltech
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/vote_ncaltech_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/vote_ncaltech_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== vote ncaltech start: $(date) ==="
/home/jnguan/.miniconda/envs/bhli/bin/python vote_ncaltech.py
echo "=== vote ncaltech end: $(date) ==="
