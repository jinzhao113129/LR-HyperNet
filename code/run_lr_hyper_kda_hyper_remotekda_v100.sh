#!/bin/bash
# LR-HyperNet P3: KDA+双曲+远程KDA扫描（EventMamba启发）
# 与P0(170805)唯一区别：remote_kda=True（远程16点按时间排序过KDA递推）
# 验证：远程点时序关系是否提供额外判别信息，预期0~1pp
#SBATCH -J lr_hyper_kda_hyper_remotekda
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_kda_hyper_remotekda_%j.out
#SBATCH -e eval_results/lr_hyper_kda_hyper_remotekda_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet KDA+Hyperbolic+RemoteKDA start: $(date) ==="
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
    --remote_kda \
    --log_dir lr_hyper_kda_hyper_remotekda \
    --log_name /lr_hyper_kda_hyper_remotekda

echo "=== done: $(date) ==="
