#!/bin/bash
#SBATCH -J secnet_baidu_150ep
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o eval_results/secnet_baidu_150ep_%j.out
#SBATCH -e eval_results/secnet_baidu_150ep_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli

echo "=== SECNet Baseline (Baidu Netdisk official data) start: $(date) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_frequency_ncaltech.py \
    --data_path ./ncaltech101_official_baidu/ \
    --num_point 10240 \
    --batch_size 80 \
    --epoch 150 \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --optimizer AdamW \
    --log_dir secnet_baidu_150ep_v100 \
    --log_name secnet_baidu_150ep

echo "=== done: $(date) ==="
