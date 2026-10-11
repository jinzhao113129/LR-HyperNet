#!/bin/bash
#SBATCH -J secnet_r1
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/secnet_r1.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/secnet_r1.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

cd /home/jnguan/JiangYouxin/ZhaoJin

echo "=== Job start: $(date) ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "=== Start training R1 (SECNet baseline) ==="
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency.py \
    --model fecnet \
    --num_category 101 \
    --data_path ./data/ \
    --num_point 8192 \
    --batch_size 64 \
    --epoch 150 \
    --log_dir secnet_r1 \
    2>&1

echo "=== Job end: $(date) ==="
