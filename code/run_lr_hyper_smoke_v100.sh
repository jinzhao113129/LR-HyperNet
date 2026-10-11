#!/bin/bash
#SBATCH -J lr_hyper_smoke_v
#SBATCH -o /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_smoke_v100_%j.out
#SBATCH -e /home/jnguan/JiangYouxin/ZhaoJin/eval_results/lr_hyper_smoke_v100_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /home/jnguan/JiangYouxin/ZhaoJin
echo "=== LR-HyperNet smoke V100 (1 epoch) start: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# V100-16GB 冒烟：batch=16, 1 epoch，验证 GPU 上无 NaN/OOM
/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --batch_size 16 \
    --epoch 1 \
    --num_category 101 \
    --num_point 10240 \
    --optimizer SGD \
    --learning_rate 0.001 \
    --decay_rate 0.0001 \
    --curvature_init 0.4 \
    --remote_g 16 \
    --data_path ./ncaltech101_3window/ \
    --log_dir smoke_1ep_v100_${SLURM_JOB_ID} \
    --log_name /lr_hypernet_smoke_v100_${SLURM_JOB_ID}
echo "=== LR-HyperNet smoke V100 end: $(date) ==="
