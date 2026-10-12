#!/bin/bash
# LR-HyperNet v3.2 (Lorentz+KDA) + Dropout0.1（无aug，轻正则）
# 关键组合实验：KDA时间编码 + 双曲聚合，轻正则
# 目标：突破欧氏裸基线38.22%
# 与 170335（KDA+aug+drop0.3）变量：去掉aug，drop从0.3降到0.1
#SBATCH -J lr_hyper_kda_v32_hyper_light
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_kda_v32_hyper_light_%j.out
#SBATCH -e eval_results/lr_hyper_kda_v32_hyper_light_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet KDA+Hyperbolic Light Regularization start: $(date) ==="
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
    --log_dir lr_hyper_kda_v32_hyper_light \
    --log_name /lr_hyper_kda_v32_hyper_light

echo "=== done: $(date) ==="
