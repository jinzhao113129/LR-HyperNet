#!/bin/bash
# LR-HyperNet v3.1 (Lorentz+time_enc) + 数据增强 + remote_dropout 0.3
# 目标：解决双曲过拟合（60轮无增强版 Train 73.5% vs Test 35.1%）
# 数据：百度官方单窗 + augment×2（3596→10788）
# 与 168575（无增强无dropout）唯一变量：augment + remote_dropout
#SBATCH -J lr_hyper_v31_aug_drop
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_v31_aug_dropout_v100_%j.out
#SBATCH -e eval_results/lr_hyper_v31_aug_dropout_v100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet v3.1 + Aug + Dropout0.3 start: $(date) ==="
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
    --time_enc \
    --curvature_init 0.4 \
    --augment 2 \
    --remote_dropout 0.3 \
    --log_dir lr_hyper_v31_aug_dropout_v100 \
    --log_name /lr_hyper_v31_aug_drop

echo "=== done: $(date) ==="
