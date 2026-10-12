#!/bin/bash
#SBATCH -J secnet_baidu_a100
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/secnet_baidu_a100_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/secnet_baidu_a100_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=gpu02

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== SECNet Baseline (Baidu data, A100-40GB, b32, 150ep) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --batch_size 32 \
    --epoch 150 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_official_baidu/ \
    --log_dir secnet_baidu_a100 \
    --log_name secnet_baidu_a100
echo "=== SECNet Baseline A100 end: $(date) ==="
