#!/bin/bash
# 消融 A1: 欧氏 + KDA（KDA单独贡献）
# 与A0同配置，仅time_enc换成kda
#SBATCH -J abl_A1_eucl_kda
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02
#SBATCH -o eval_results/abl_A1_eucl_kda_%j.out
#SBATCH -e eval_results/abl_A1_eucl_kda_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== Ablation A1 (Euclidean + KDA) start: $(date) ==="
echo "Host: $(hostname)"

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
    --kda \
    --remote_dropout 0.1 \
    --seed 42 \
    --log_dir abl_A1_eucl_kda \
    --log_name /abl_A1_eucl_kda
