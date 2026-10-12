#!/bin/bash
#SBATCH -J lr_hyper_kda_v32_sanity
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o eval_results/lr_hyper_kda_v32_sanity_%j.out
#SBATCH -e eval_results/lr_hyper_kda_v32_sanity_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin

/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --epoch 50 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --log_dir lr_hyper_kda_v32_sanity \
    --hyperbolic_type lorentz \
    --curvature_init 0.4 \
    --kda \
    --sanity_5class \
    --group_norm \
    --grad_accum 4
