#!/bin/bash
#SBATCH -J sn_fu_base_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/sn_fu_base_ai2_%j.out
#SBATCH -e eval_results/sn_fu_base_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-15 00:55 ai2: 全量数据基线（学长指示：直接对标官方协议全量全轮数）
# 数据 = ncaltech101_official_gen（官方脚本产物 62703/5226，共识 §7.2）
# 配置 = 官方口径：AdamW lr=1e-3 wd=1e-4 + Cosine + 150 epoch
# V100-32GB 单卡 batch 16 + grad_accum 2（等效 32，接近官方"吃满显存"口径）
# 12h 时限跑不满 150ep → checkpoint 续训（脚本自动 resume）
# KNN 已按 Codex 必改项换回官方全文 argsort（modules.py md5 219b90bc）
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_gen/ \
    --num_point 10240 \
    --batch_size 16 \
    --epoch 150 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_full_baseline_s42_ai2 \
    --log_name /secnet_full_baseline_s42_ai2 \
    --grad_accum 2 \
    --seed 42
