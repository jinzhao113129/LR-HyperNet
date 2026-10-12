#!/bin/bash
# LR-HyperNet v3.2 (Lorentz+KDA) + Aug + Dropout0.3
# 组合实验：KDA时间编码 + 双曲聚合（两个都有效的组件组合）
# 与 170335（KDA+欧氏聚合）唯一变量：--hyperbolic_type lorentz
# 与 170149（双曲+time_enc）唯一变量：--kda（替换--time_enc）
#SBATCH -J lr_hyper_kda_v32_hyper
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_kda_v32_hyper_v100_%j.out
#SBATCH -e eval_results/lr_hyper_kda_v32_hyper_v100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet v3.2 KDA + Hyperbolic + Aug + Dropout0.3 start: $(date) ==="
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
    --augment 2 \
    --remote_dropout 0.3 \
    --log_dir lr_hyper_kda_v32_hyper_v100 \
    --log_name /lr_hyper_kda_v32_hyper

echo "=== done: $(date) ==="
