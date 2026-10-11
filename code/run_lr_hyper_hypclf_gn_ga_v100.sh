#!/bin/bash
#SBATCH -J lr_hyper_hypclf_gn_ga
#SBATCH -p gpu
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o log/slurm-%j.out
#SBATCH -e log/slurm-%j.err
#SBATCH --exclude=gpu02

# 路线A：欧氏backbone + 双曲原型分类头
# 与 168574（欧氏+GN+GA）同配置，唯一变量是分类头换成双曲原型分类器
# 目的：验证双曲增益是否出现在分类层而非聚合层

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
    --log_dir lr_hyper_hypclf_gn_ga_v100 \
    --log_name /lr_hyper_hypclf_gn_ga \
    --hyperbolic_type euclidean \
    --hyperbolic_classifier \
    --time_enc \
    --group_norm \
    --grad_accum 4 \
    --curvature_init 0.4
