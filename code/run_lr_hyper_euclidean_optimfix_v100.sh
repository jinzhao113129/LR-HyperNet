#!/bin/bash
# LR-HyperNet 欧氏基线 + 优化器修复（验证优化器是否让模型学得动）
# 数据：百度官方单窗 ncaltech101_official_baidu（train 3596 / test 934，101 类）
# 修复：group_norm + grad_accum 4（去掉 balanced_sampler 和 logit_adj，先确认学得动）
# 目标：Train >90%（证明欠拟合是 BN+b8 问题），作为新对照基线
# 卡：V100-16GB（gpu11 空闲）
#SBATCH -J lr_hyper_eucl_optimfix
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_eucl_optimfix_v100_%j.out
#SBATCH -e eval_results/lr_hyper_eucl_optimfix_v100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet Euclidean + OptimFix start: $(date) ==="
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
    --hyperbolic_type euclidean \
    --time_enc \
    --log_dir lr_hyper_eucl_gn_ga_v100 \
    --log_name /lr_hyper_eucl_gn_ga

echo "=== done: $(date) ==="
