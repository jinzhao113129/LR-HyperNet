#!/bin/bash
# LR-HyperNet v3.1 V100 sanity check（快速验证，不做正式对比）
# v3.1 = time_enc(修复版) + 曲率优化三件套 + Lorentz
#SBATCH -J lr_hyper_v31_v100_sanity
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 06:00:00
#SBATCH --exclude=gpu02
#SBATCH -o log/slurm-%j.out
#SBATCH -e log/slurm-%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PATH=/home/jnguan/.miniconda/envs/bhli/bin:$PATH

python train_lr_hyper.py \
    --batch_size 8 \
    --epoch 30 \
    --num_point 10240 \
    --data_path ./ncaltech101_official_baidu/ \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 1e-4 \
    --curvature_init 0.4 \
    --remote_g 16 \
    --time_enc \
    --hyperbolic_type lorentz \
    --log_dir lr_hyper_v31_v100_sanity \
    --log_name /lr_hyper_v31_v100_sanity
