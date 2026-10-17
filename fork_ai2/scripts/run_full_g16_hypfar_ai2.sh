#!/bin/bash
#SBATCH -J sn_g16_hypfar_s42
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_g16_hypfar_s42_%j.out
#SBATCH -e eval_results/sn_g16_hypfar_s42_%j.err

# 2026-10-15 ai2: G16 六处理训练链（hyp_far = 双曲距离选 16 远程点）。
#
# 口径（Codex 第五轮审查要求写清，单一事实源）：
#   - 骨干：LRHyperNet 默认（无 kda/group_norm/dropout/anchor）——
#     唯一在官方全量数据上健康收敛的骨干（v1 默认 ep8 已 56.3% seq 斜率健康）；
#     这是"唯一变量是 selector 的干净对照"，不混入 kda/双曲聚合等其他维度。
#   - 数据：ncaltech101_official_gen（train 62703 / test 5226 × 10240 点）
#   - epoch 150 + AdamW lr=1e-3 + Cosine（train_lr_hyper.py 内置 scheduler）
#   - batch 32 + grad_accum 2（等效 batch 64，与主线 FECNet 全量链同口径）
#   - remote_g=16（G16 核心，LRHyperNet 默认），curvature_init=0.4（默认）
#   - 六处理唯一差异 = --g16_treatment；hyp_far 内部把 selector 切到
#     lorentz 双曲距离选点（local_remote_grouper_v2 的 treatment 映射），
#     命令行不传 --hyperbolic_type（两条链该 flag 均保持默认 poincare，
#     hyper_agg 聚合流形两条相同——只有 selector 选点准则不同）。
#   - --g16_profile --g16_profile_p 0.01：训练时 Bernoulli 采样落选择分布
#     诊断（log/lr_hypernet/<log_dir>/g16_profile.h5），seed 42 固定。
#   - --augment 2（2026-10-17 加，用户决策"以后跑的所有都做数据增强"）：
#     每训练样本×3 增强（crop/reverse/极性翻转/抖动/丢点），test 不增强。
#     train_lr_hyper.py 原生支持该 flag。预算影响：每 epoch 慢 ~3 倍
#     （52min/ep），30h 时限内 ~35ep——早期轨迹对比 + KDA 式接力续跑。
#
# 时限：LRHyperNet 官方全量 150ep 从未跑完过（历史全 CANCELLED），
#   比 FECNet（51.5h）多 remote 选点+双曲聚合，估 60-70h，给 72h 缓冲。
#   --no-requeue：防抢占自动重提读到未来新代码（续跑走人工 KDA 式 resume）。
#
# 点数（Codex P0 修正：预扣口径，不是实时消耗）：
#   A800 ≈ 16.2 点/h（按当前预扣），72h 单条 ≈ 1166.4 点，两条 ≈ 2332.8 点。
#   提交前必须 `scir-account details` 的 Avail ≥ 1166.4（第一条）；
#   `scir-account -d` 的 Avail 是未扣排队预扣的实时消耗值，不可用。
#   预扣制下排队也占余额——余额不足时不要为了排队提前 sbatch。
#
# 与 euc_near 链唯一差异：-J / --log_dir / --log_name / --g16_treatment。
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH

/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --data_path ./ncaltech101_official_gen/ \
    --num_point 10240 \
    --batch_size 32 \
    --grad_accum 2 \
    --epoch 150 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --remote_g 16 \
    --curvature_init 0.4 \
    --log_dir lr_hyper_g16_hypfar_s42_ai2 \
    --log_name /lr_hyper_g16_hypfar_s42_ai2 \
    --g16_profile \
    --g16_profile_p 0.01 \
    --g16_treatment hyp_far \
    --augment 2 \
    --ema 0.999 \
    --seed 42
