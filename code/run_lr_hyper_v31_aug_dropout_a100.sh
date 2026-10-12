#!/bin/bash
# LR-HyperNet v3.1 (Lorentz+time_enc) + 数据增强 + remote_dropout 0.3
# 卡：A100 40GB PCIe（gpu16 有 2 张可用）
#SBATCH -J lr_hyper_v31_aug_drop
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o eval_results/lr_hyper_v31_aug_dropout_a100_%j.out
#SBATCH -e eval_results/lr_hyper_v31_aug_dropout_a100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet v3.1 + Aug + Dropout0.3 (A100) start: $(date) ==="
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
    --log_dir lr_hyper_v31_aug_dropout_a100 \
    --log_name /lr_hyper_v31_aug_drop_a100

echo "=== done: $(date) ==="
