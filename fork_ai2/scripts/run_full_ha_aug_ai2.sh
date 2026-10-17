#!/bin/bash
#SBATCH -J sn_fu_ha_aug_s42
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 30:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_fu_ha_aug_s42_%j.out
#SBATCH -e eval_results/sn_fu_ha_aug_s42_%j.err

# 2026-10-17 早上 ai2: 双曲注意力 + 数据增强 + EMA 链（方案 B 强化版）。
# - 与已停的 ha 纯版（176974）训练 flag 差异：--augment 2 + --ema 0.999
#   （用户决策：跑之前的工作做到极致，没开跑/刚开跑的链全部加满）。
# - EMA（2026-10-17 接线）：checkpoint 额外存 net_ema 键，best/last 判定
#   不变（不污染对照），EMA 是并行出口（文献 +0.5~1.5pp 小数据集更稳）。
# - 用户决策链：① 175245 选点链已收敛止损（best 70.36%/72.56% seq）
#   ② 后续所有冲刺链加增强 ③ 本链选 B 方案（注意力+增强，不带选点
#   ——与选点链的对照保持"组件差"单一变量：选点 vs 注意力+增强）
# - 配方：official_gen / 10240 点 / batch 32 + grad_accum 2 /
#   AdamW lr=1e-3 + Cosine / 150ep / seed 42
# - 时限 30h：数据×3 后每 epoch 慢 ~3 倍（~52min/ep），150ep 需 ~130h
#   【跑不完】——本链 30h 只能跑到 ~35ep。策略：30h 拿增强版早期轨迹
#   （对比无增强 ha 链同期），终数靠接力（KDA 式 resume，checkpoint
#   自动续）或 11 月额度恢复后提长版。scir 预扣 480 点（余额 498 内）。
# - --no-requeue：续跑走人工 KDA 式（防 requeue 读到未来新代码）
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
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
    --log_dir secnet_full_ha_aug_s42_v2_ai2 \
    --log_name /secnet_full_ha_aug_s42_v2_ai2 \
    --grad_accum 2 \
    --augment 2 \
    --ema 0.999 \
    --seed 42 --hyperbolic_attention
