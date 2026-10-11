#!/bin/bash
#SBATCH -J lr_hyper_eucl_gn_ga_aug
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o log/slurm-%j.out
#SBATCH -e log/slurm-%j.err
#SBATCH --exclude=gpu02

# 欧氏backbone + 数据增强（augment×2 → 3596×3=10788）
# 与 168574（欧氏+GN+GA，无增强）对比：验证数据增强是否解决欠拟合/提升泛化
# 无增强版本 Train 31%@epoch16，增强后数据×3，期待 Train 更快爬升

cd /home/jnguan/JiangYouxin/ZhaoJin

/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 60 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 1e-4 \
    --num_category 101 \
    --log_dir lr_hyper_eucl_gn_ga_aug_v100 \
    --log_name /lr_hyper_eucl_gn_ga_aug \
    --hyperbolic_type euclidean \
    --time_enc \
    --group_norm \
    --grad_accum 4 \
    --augment 2 \
    --curvature_init 0.4
