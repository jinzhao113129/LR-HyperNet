#!/bin/bash
#SBATCH -J g16_fullmem_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --exclude=gpu02,gpu12,gpu15,gpu18
#SBATCH -o /home/jnguan/eval_results/g16_fullmem_ai2_%j.out
#SBATCH -e /home/jnguan/eval_results/g16_fullmem_ai2_%j.err

# 2026-10-15 12:40 ai2/Claude: 环境修复重提。
# 问题: bhli 的 torch(2.3.1+cu121) 链接 libcudnn.so.8, 但 pip 的
# nvidia-cudnn-cu12 已被升到 9.5.1.17(只有 .so.9)——torch import 即崩
# (共享环境被人升级弄坏, 不是脚本问题; 昨天 173720 成功时 .8 还在)。
# 修复: 从用户垃圾桶里残留的旧 venv 拷出完整 cudnn8 库(1.2GB, 7 个 .so.8)
# 到 ai2_g16/cudnn8_lib/, 本脚本 LD_LIBRARY_PATH 指过去——不动共享 bhli。
cd /home/jnguan/JiangYouxin/ZhaoJin
source /home/jnguan/.miniconda/etc/profile.d/conda.sh
conda activate bhli
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH

echo "=== G16 full model fwd+bwd mem test: $(date) ==="
md5sum lr_hypernet.py models/local_remote_grouper.py

python /home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/test_g16_full_model_mem.py

echo "=== done $(date) rc=$? ==="
