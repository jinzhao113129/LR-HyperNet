#!/bin/bash
# LR-HyperNet v3.1 (Lorentz+time_enc) + Aug + Dropout0.1
# 调参实验：对比dropout 0.3（170149 Test 30.62%），看0.1是否更合适
# 与 170149 唯一变量：--remote_dropout 0.1
#SBATCH -J lr_hyper_v31_aug_drop01
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_v31_aug_drop01_v100_%j.out
#SBATCH -e eval_results/lr_hyper_v31_aug_drop01_v100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet v3.1 + Aug + Dropout0.1 start: $(date) ==="
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
    --remote_dropout 0.1 \
    --log_dir lr_hyper_v31_aug_drop01_v100 \
    --log_name /lr_hyper_v31_aug_drop01

echo "=== done: $(date) ==="
