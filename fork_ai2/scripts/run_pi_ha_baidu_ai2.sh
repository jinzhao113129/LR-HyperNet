#!/bin/bash
#SBATCH -J sn_pi2_ha_baidu
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_pi_ha_baidu_%j.out
#SBATCH -e eval_results/sn_pi_ha_baidu_%j.err

# 2026-10-17 ai2: 小样本快测链（用户决策回归"先小样本正反馈→再全量"）。
# 协议：
#   - 数据：ncaltech101_official_baidu（train 3596 / test 934，3596/934 已
#     在 HPC 校验）——P0 时代的小样本协议（注意：当时判"坏基线"，但用户
#     指令明确回归此协议做快测，风险已知情：小样本的 acc 数字不作结论，
#     只看"组件 vs 基线的配对差方向"）；
#   - 满轮数 = 收敛即停：--patience 20（连续 20 epoch best seq 无提升
#     即正常退出），--epoch 300 上限（小样本收敛快，实际预计 60-120ep）；
#   - 组件：双曲注意力（ha）+ 增强 + EMA（全套配方不变，只是小数据）；
#   - 对照锚：同协议无组件基线（run_pi_base_baidu，配对差 = 组件净增益）。
#   - batch 16 + grad_accum 2（V100-16 实测配方）。
# 时限 12h（小样本每 epoch ~3min，300ep 上限 15h——patience 会先触发）。
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
    --log_dir secnet_pi2_ha_baidu_ai2 \
    --log_name /secnet_pi2_ha_baidu_ai2 \
    --augment 2 \
    --ema 0.999 \
    --seed 42 --hyperbolic_attention
