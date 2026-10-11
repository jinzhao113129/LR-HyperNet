#!/bin/bash
# LR-HyperNet v3.1 修复版 V100 验证（Lorentz 数值修复后）
# 修复：lorentz_exp_origin ①加 1/√c 因子（流形约束）②范数 clamp≤10（防 cosh 溢出 + 内积减法灾难）
# 与失败的 168078 同配置（time_enc + lorentz + 曲率优化，b8 30ep），直接对比
#SBATCH -J lr_hyper_v31_fix_v100
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 10:00:00
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
    --log_dir lr_hyper_v31_fix_v100 \
    --log_name /lr_hyper_v31_fix_v100
