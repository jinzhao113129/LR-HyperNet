#!/bin/bash
# 消融 A2: Lorentz双曲 + time_enc（双曲单独贡献）
# 与A0同配置，仅euclidean换成lorentz
#SBATCH -J abl_A2_lorentz_timeenc
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu02
#SBATCH -o eval_results/abl_A2_lorentz_timeenc_%j.out
#SBATCH -e eval_results/abl_A2_lorentz_timeenc_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== Ablation A2 (Lorentz + time_enc) start: $(date) ==="
echo "Host: $(hostname)"

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
    --remote_dropout 0.1 \
    --seed 42 \
    --log_dir abl_A2_lorentz_timeenc \
    --log_name /abl_A2_lorentz_timeenc
