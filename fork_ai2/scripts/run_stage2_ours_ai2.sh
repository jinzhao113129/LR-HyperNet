#!/bin/bash
#SBATCH -J s2_ours_v1
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/stage2/logs/s2_ours_v1_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/stage2/logs/s2_ours_v1_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --no-requeue

# 2026-10-16 ai2: Stage-2 攻坚链（用户指令"接 Stage-2 超 82.4"）
# 用我们自己 175245（双曲选点链）的 best checkpoint：
#   ① 特征提取（修复版 load_backbone：从 ckpt.args 恢复 14 组件 flag）
#   ② 自训 Stage-2 跨窗分类器（官方配方：SGD lr=0.1 momentum=0.9 wd=1e-4 + cosine）
#   ③ 评估输出 Stage-1 四基线 + Stage-2 终局 seq acc —— 直接对 82.4
#
# 口径：
#   - 175245 = --hyperbolic_selector 训练（load_backbone 已实测正确恢复
#     HyperbolicUniformGrouper，前 300 窗单窗 acc 70.33% ≈ 训练日志 70.28%）
#   - RNG：每 split 固定重播种（test=42/train=43，Codex 修复版），
#     特征与调用顺序无关、可复现
#   - 官方 Stage-2 权重已证不适用（套我们的特征=1% 随机），走 --init scratch
#   - forward_mode 主假设 attn_concat（attention 加权 3 窗拼接 3240 维）
#   - V100-16（4 点/h × 6h = 24 点；实际特征提取~3h+训练~1h，4h 内跑完）
#
# 口径（Codex 第三轮非阻断#2 诚实声明）：
#   - Stage-1 特征来自 175245 的 best_checkpoint.pth，而 175245 自己的
#     best 是按测试集 seq acc 选出的（test-select 偏置）——
#     最终口径 = "Stage-1 best-on-test + Stage-2 last/val-selected"，
#     不是完全干净；论文若主张完全干净需补 Stage-1 last 的对应矩阵。
#   - Stage-2 侧已消除自己的 best-on-test（val_ratio=0.1 验证集选 best，
#     last 为主口径）。
# 提交时机：175245 跑完出 best checkpoint 后（预计 10/17 下午/晚上）
# —— 现在提交会在 checkpoint 还在更新时提取，特征口径不干净。
cd /home/jnguan/JiangYouxin/ZhaoJin/stage2
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH
# Codex P1 修复：失败即停（特征提取 OOM/报错时不继续训练评估，避免半成品）
set -e
# Codex P1 修复：每次运行用带时间戳的独立输出目录（重跑不误读旧文件/半成品）
RUN_TS=$(date +%Y%m%d_%H%M%S)
FEAT_DIR=/home/jnguan/JiangYouxin/ZhaoJin/stage2/features_ours_${RUN_TS}
TRAIN_DIR=/home/jnguan/JiangYouxin/ZhaoJin/stage2/log_stage2_ours_${RUN_TS}
CKPT=/home/jnguan/JiangYouxin/ZhaoJin/log/classification/secnet_full_hsel_s42_ai2/checkpoints/best_checkpoint.pth

echo "=== [1/3] 特征提取（175245 best ckpt）start: $(date) ==="
/home/jnguan/.miniconda/envs/bhli/bin/python stage2_extract_features.py \
    --data_path /home/jnguan/JiangYouxin/ZhaoJin/ncaltech101_official_gen/ \
    --ckpt "$CKPT" \
    --split both --batch 16 \
    --out_dir "$FEAT_DIR"
echo "=== 提取完成: $(date) ==="

echo "=== [2/3] 自训 Stage-2 跨窗分类器 start: $(date) ==="
/home/jnguan/.miniconda/envs/bhli/bin/python stage2_train.py \
    --train_features "$FEAT_DIR/features_train.h5" \
    --test_features "$FEAT_DIR/features_test.h5" \
    --out_dir "$TRAIN_DIR" \
    --forward_mode attn_concat \
    --sample_mode triple \
    --init scratch \
    --val_ratio 0.1 \
    --epochs 100 --batch 128 --lr 0.1 --seed 42
echo "=== 训练完成: $(date) ==="

echo "=== [3/3] 最终评估（Codex P0 双口径）: $(date) ==="
# last=主口径（干净，训练终点无 test-select 偏置）；best=exploratory（标注偏置）
/home/jnguan/.miniconda/envs/bhli/bin/python stage2_eval.py \
    --features "$FEAT_DIR/features_test.h5" \
    --ckpt "$TRAIN_DIR" \
    --which both
echo "=== Stage-2 攻坚链完成: $(date)（last 口径 vs 82.4 = 对 82.4 的干净差距） ==="
