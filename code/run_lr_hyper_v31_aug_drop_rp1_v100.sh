#!/bin/bash
# LR-HyperNet v3.1 (Lorentz+time_enc) + Aug + Dropout0.3 + RP1空间新颖性选择器
# 消融实验：验证RP1 Selector（空间新颖性远程点选择）是否有效
# 与 170149（双曲+aug+drop）唯一变量：--use_rp1_selector
#SBATCH -J lr_hyper_v31_aug_drop_rp1
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_v31_aug_drop_rp1_v100_%j.out
#SBATCH -e eval_results/lr_hyper_v31_aug_drop_rp1_v100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet v3.1 + Aug + Dropout0.3 + RP1 Selector start: $(date) ==="
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
    --use_rp1_selector \
    --rp1_lambda 1.0 \
    --log_dir lr_hyper_v31_aug_drop_rp1_v100 \
    --log_name /lr_hyper_v31_aug_drop_rp1

echo "=== done: $(date) ==="
