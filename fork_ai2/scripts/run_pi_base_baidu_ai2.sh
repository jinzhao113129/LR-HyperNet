#!/bin/bash
#SBATCH -J sn_pi2_base_baidu
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_pi_base_baidu_%j.out
#SBATCH -e eval_results/sn_pi_base_baidu_%j.err

# 2026-10-17 ai2: 小样本快测【基线对照】链。
# 与 run_pi_ha_baidu_ai2.sh 逐 flag 一致，唯一差异 = 不带组件 flag
# （无 --hyperbolic_attention）。配对差 = 双曲注意力+配方 在小样本上的
# 净增益方向。协议同那条：baidu 3596/934、patience 20 收敛即停、
# augment 2 + EMA 同开（配方变量固定，只测组件）。
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH

/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 16 \
    --grad_accum 2 \
    --epoch 300 \
    --patience 20 \
    --early_stop_metric acc \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_pi2_base_baidu_ai2 \
    --log_name /secnet_pi2_base_baidu_ai2 \
    --augment 2 \
    --ema 0.999 \
    --seed 42
