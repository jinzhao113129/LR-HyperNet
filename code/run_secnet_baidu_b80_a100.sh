#!/bin/bash
# SECNet 官方基线复现 — b80（对齐官方 batch_size=80，排除 batch 变量）
# 数据：百度网盘官方单窗 ncaltech101_official_baidu（train 3596 / test 934，101 类）
# 目标：逼近官方 82.4%；此前 b32 跑出 Train 99.86% / Test 54.07%（过拟合）
# 卡：A100-40GB（gpu09 有空位）
#SBATCH -J secnet_baidu_b80
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02
#SBATCH -o eval_results/secnet_baidu_b80_a100_%j.out
#SBATCH -e eval_results/secnet_baidu_b80_a100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== SECNet Baseline b80 (Baidu official data) start: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 80 \
    --epoch 150 \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --optimizer AdamW \
    --log_dir secnet_baidu_b80_a100 \
    --log_name secnet_baidu_b80

echo "=== done: $(date) ==="
