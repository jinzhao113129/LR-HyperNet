#!/bin/bash
# SECNet 基线 b8/60ep 百度单窗数据
# 目的：与消融基线 168574（LRHyperNet欧氏版 b8/60ep=38.22%）同配置对比
# 回答：16点差距是batch/epoch导致，还是远程聚合结构本身有问题
# 与 168059（b32/150ep=54.07%）唯一区别：batch 8、epoch 60
#SBATCH -J secnet_baidu_b8_60ep
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 12:00:00
#SBATCH --exclude=gpu02
#SBATCH -o log/slurm-%j.out
#SBATCH -e log/slurm-%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PATH=/home/jnguan/.miniconda/envs/bhli/bin:$PATH

python train_frequency_ncaltech.py \
    --batch_size 8 \
    --epoch 60 \
    --num_point 10240 \
    --data_path ./ncaltech101_official_baidu/ \
    --optimizer AdamW \
    --learning_rate 0.001 \
    --decay_rate 1e-4 \
    --log_dir secnet_baidu_b8_60ep \
    --log_name /secnet_baidu_b8_60ep
