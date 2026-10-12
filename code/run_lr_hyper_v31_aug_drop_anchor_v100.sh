#!/bin/bash
# 步骤5：双曲+aug+drop0.3 + 全局锚点（use_global_anchor）
# 在步骤3（170112）基础上加PVAG启发可学习全局锚点
#SBATCH -J lr_hyper_v31_aug_drop_anchor
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o eval_results/lr_hyper_v31_aug_drop_anchor_v100_%j.out
#SBATCH -e eval_results/lr_hyper_v31_aug_drop_anchor_v100_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== v3.1 + Aug + Drop + GlobalAnchor start: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_lr_hyper.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 8 \
    --grad_accum 4 \
    --group_norm \
    --epoch 60 \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --optimizer AdamW \
    --hyperbolic_type lorentz \
    --time_enc \
    --curvature_init 0.4 \
    --augment 2 \
    --remote_dropout 0.3 \
    --use_global_anchor \
    --anchor_lambda 0.5 \
    --log_dir lr_hyper_v31_aug_drop_anchor_v100 \
    --log_name /lr_hyper_v31_aug_drop_anchor

echo "=== done: $(date) ==="
