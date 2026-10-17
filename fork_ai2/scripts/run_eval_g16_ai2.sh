#!/bin/bash
# G16 冲刺链评测（LRHyperNet 版推理）
#SBATCH -J sn_g16_eval
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_g16_eval_%j.out
#SBATCH -e eval_results/sn_g16_eval_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
set -e

# LRHyperNet 专用推理（tools/extract_preds_lrh.py，log/lr_hypernet 路径约定）
# raw/EMA 双出口（scheme=step 时自动）
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds_lrh.py \
    --log_dir lr_hyper_g16_hypfar_s42_ai2 \
    --which both \
    --data_path ./ncaltech101_official_gen/ \
    --batch_size 32 \
    --eval_seed 42

echo "=== 评测完成，npz 清单 ==="
ls -la log/lr_hypernet/lr_hyper_g16_hypfar_s42_ai2/*seed42.npz 2>/dev/null
