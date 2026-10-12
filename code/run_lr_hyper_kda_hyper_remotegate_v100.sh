#!/bin/bash
# LR-HyperNet P2: KDA+双曲+远程分支门控融合（SECNet启发）
# 与P0(170805)唯一区别：remote_gate=True（中心点特征门控调制远程特征）
# 验证：乘性软调制vs加性硬注入，预期0~1pp+训练稳定性
#SBATCH -J lr_hyper_kda_hyper_remotegate
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_kda_hyper_remotegate_%j.out
#SBATCH -e eval_results/lr_hyper_kda_hyper_remotegate_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet KDA+Hyperbolic+RemoteGate start: $(date) ==="
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
    --remote_gate \
    --log_dir lr_hyper_kda_hyper_remotegate \
    --log_name /lr_hyper_kda_hyper_remotegate

echo "=== done: $(date) ==="
