#!/bin/bash
#SBATCH -J hfecnet_a100
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/hfecnet_a100_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/hfecnet_a100_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== HFECNet A100 full repro start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# 官方大配置 + 3窗口数据 + 与 SECNet 基线同协议
# batch=80 (A100-80GB 上限，与 SECNet 基线对齐)
# epoch=350, SGD lr=0.001 wd=0.0001 (官方训练日志)
# num_point=10240 (与生成数据一致)
/home/jnguan/.miniconda/envs/bhli/bin/python train_hfecnet.py \
    --batch_size 80 \
    --epoch 350 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer SGD \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --data_path ./ncaltech101_3window/ \
    --log_dir hfecnet_a100_3window \
    --ablation none
echo "=== HFECNet A100 full repro end: $(date) ==="
