#!/bin/bash
#SBATCH -J lr_hyper_v3_small_baidu
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_small_baidu_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v3_small_baidu_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet v3 SMALL (KDA, 30ep, b16, baidu data, no global_anchor) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 16 \
    --epoch 30 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_baidu/ \
    --log_dir lr_hyper_v3_small_baidu \
    --log_name /lr_hyper_v3_small_baidu \
    --remote_g 16 \
    --curvature_init 0.4 \
    --kda
echo "=== LR-HyperNet v3 SMALL end: $(date) ==="
