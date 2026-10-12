#!/bin/bash
# LR-HyperNet P0 + Random选点（原文RP1基线）
# 基于P0配置（KDA+双曲+drop0.1无aug），远程点选择改为随机
# 目的：验证16个点随机选是否比特征距离topk差，作为选点消融最低基线
#SBATCH -J lr_hyper_kda_hyper_random
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_kda_hyper_random_%j.out
#SBATCH -e eval_results/lr_hyper_kda_hyper_random_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet KDA+Hyper+Random Selector start: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_lr_hyper.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --grad_accum 4 \
    --group_norm \
    --epoch 60 \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --optimizer AdamW \
    --hyperbolic_type lorentz \
    --kda \
    --curvature_init 0.4 \
    --remote_dropout 0.1 \
    --selector_mode random \
    --log_dir lr_hyper_kda_hyper_random \
    --log_name /lr_hyper_kda_hyper_random

echo "=== done: $(date) ==="
