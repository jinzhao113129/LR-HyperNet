#!/bin/bash
# LR-HyperNet P4: KDA+双曲+密度输入通道（SpikePoint启发）
# 与P0(170805)唯一区别：density_channel=True（32x32体素log压缩密度通道）
# 验证：事件密度信息（边缘/运动vs噪声）是否帮助分类，预期0~0.5pp
#SBATCH -J lr_hyper_kda_hyper_density
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o eval_results/lr_hyper_kda_hyper_density_%j.out
#SBATCH -e eval_results/lr_hyper_kda_hyper_density_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== LR-HyperNet KDA+Hyperbolic+Density start: $(date) ==="
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
    --density_channel \
    --density_bins 32 \
    --log_dir lr_hyper_kda_hyper_density \
    --log_name /lr_hyper_kda_hyper_density

echo "=== done: $(date) ==="
