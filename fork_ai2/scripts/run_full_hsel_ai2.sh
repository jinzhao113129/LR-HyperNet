#!/bin/bash
#SBATCH -J sn_fu_hsel_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_fu_hsel_ai2_%j.out
#SBATCH -e eval_results/sn_fu_hsel_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-15 00:55 ai2: 全量数据 + 双曲距离选 16 个远程点（学长核心创新方向，
# use_hyperbolic_selector 历史已有 flag——Poincaré 网格选点，与 P0 的
# hsel_fix 同口径。注意：这是 HyperbolicUniformGrouper 路径（中心点层面
# 的双曲网格），G16 RemoteSelector 的 hyp_far 训练接线仍等 Codex 审批）
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_gen/ \
    --num_point 10240 \
    --batch_size 16 \
    --epoch 150 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_full_hsel_s42_ai2 \
    --log_name /secnet_full_hsel_s42_ai2 \
    --grad_accum 2 \
    --hyperbolic_selector \
    --seed 42
