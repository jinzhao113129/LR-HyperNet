#!/bin/bash
#SBATCH -J sn_fu_hsel_r1
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 2-08:00:00
#SBATCH -o eval_results/sn_fu_hsel_r1_%j.out
#SBATCH -e eval_results/sn_fu_hsel_r1_%j.err

# 2026-10-15 12:50 ai2: KDA 式续跑（用户指出集群时限真相后的简化方案）。
# - compute 分区 MaxTime=7-00:00:00（7天）——原 12h 假设错误；
# - 150ep 全程 ~48.5h < 2 天，单作业可跑完，无需 v5 链式编排；
# - 脚本自动检测 last_checkpoint（p1 已跑 ~26ep）→ 从 epoch+1 续；
# - 配置与 p1 完全一致（resume 校验 20+ 键全对齐才能过）。
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
# 2026-10-15 15:20 补丁: bhli 的 cudnn 被升级弄坏(torch 要 .8, pip 装的是 .9)
# p1 凌晨启动躲过了; 175233 因此 FAILED。指向我们自备的 cudnn8 库(不动共享 bhli)
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH

/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_gen/ \
    --num_point 10240 \
    --batch_size 32 \
    --epoch 150 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --log_dir secnet_full_hsel_s42_ai2 \
    --log_name /secnet_full_hsel_s42_ai2 \
    --grad_accum 2 \
    --seed 42 --hyperbolic_selector
