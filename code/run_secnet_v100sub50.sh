#!/bin/bash
#SBATCH -J secnet_v100sub50
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/secnet_v100sub50_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/secnet_v100sub50_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== SECNet baseline V100 SUB50 start: $(date) ==="
echo "Host: $(hostname)"
# 小版本：train 25% 子集 (~15.7k) + test 全量 5226 + 50ep，仅趋势/sanity，不进论文
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --batch_size 16 \
    --epoch 30 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_sub/ \
    --log_dir secnet_official_gen_sub50_v100b16 \
    --log_name /secnet_v100sub50
echo "=== SECNet baseline V100 SUB50 end: $(date) ==="
