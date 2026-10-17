#!/bin/bash
#SBATCH -J sn_fu_ha_s42
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 2-16:00:00
#SBATCH -o eval_results/sn_fu_ha_ai2_%j.out
#SBATCH -e eval_results/sn_fu_ha_ai2_%j.err

# 2026-10-15 13:00 ai2: 全量双曲注意力组件链（第一梯队第二位）。
# - 与双曲选点链唯一差异: --hyperbolic_attention（Lorentz 双曲距离自注意力
#   残差支路，P0 单 seed 最强单组件 +3.0pp p=0.0097）
# - 数据/训练配置与双曲选点链完全一致（同协议对照）:
#   ncaltech101_official_gen / 10240 点 / batch 32 + grad_accum 2 /
#   AdamW lr=1e-3 + Cosine / 150ep / seed 42
# - log_dir: secnet_full_ha_s42_ai2（与选点链 secnet_full_hsel_s42_ai2
#   平行命名——组件对照关系一眼可读）
# - 时限 64h（51.5h 需求 + 24% 缓冲，单作业跑完；KDA 式自动 resume 兜底）
# - 唯一对照锚: 双曲选点链终数（同协议同 seed 唯一差异=组件 flag）
#   → 配对差 = 双曲注意力 vs 双曲选点 的真实增益
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
# 2026-10-16 22:50 补丁: bhli cudnn8 修复（175251 首跑因此 FAILED：
# torch 要 libcudnn.so.8，共享环境只有 .9——指向自备 cudnn8_lib，
# 与选点链 resume 脚本同一修法）
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
    --log_dir secnet_full_ha_s42_ai2 \
    --log_name /secnet_full_ha_s42_ai2 \
    --grad_accum 2 \
    --seed 42 --hyperbolic_attention
