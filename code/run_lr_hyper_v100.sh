#!/bin/bash
#SBATCH -J lr_hyper_v100
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v100_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_v100_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet V100 full training start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# V100-16GB 正式训练：batch=16, 350ep
# 与 SECNet 基线（V100 版）完全同协议：10240点, SGD lr=0.001 wd=1e-4
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 16 \
    --epoch 350 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer SGD \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --curvature_init 0.4 \
    --remote_g 16 \
    --data_path ./ncaltech101_3window/ \
    --log_dir lr_hypernet_350ep_v100 \
    --log_name /lr_hypernet_350ep_v100
echo "=== LR-HyperNet V100 full training end: $(date) ==="
