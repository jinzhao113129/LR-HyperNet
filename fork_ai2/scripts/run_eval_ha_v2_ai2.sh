#!/bin/bash
# ha v2 冲刺链评测（checkpoint 出现后手动/自动触发）
# Codex 提醒：FECNet 评测必须传 --data_path ncaltech101_official_gen --batch_size 32
# （脚本历史默认是 baidu/batch 8，会错数据）
#SBATCH -J sn_ha_eval
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_ha_eval_%j.out
#SBATCH -e eval_results/sn_ha_eval_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
set -e

# 双口径（best+last）× raw/EMA 双出口（net_ema 存在且 scheme=step 时自动评 EMA）
# 输出对齐 mcnemar：log/classification/<log_dir>/<which>[_ema]_seed42.npz
/home/jnguan/.miniconda/envs/bhli/bin/python tools/extract_preds.py \
    --log_dir secnet_full_ha_aug_s42_v2_ai2 \
    --which both \
    --data_path ./ncaltech101_official_gen/ \
    --batch_size 32 \
    --eval_seed 42

echo "=== 评测完成，npz 清单 ==="
ls -la log/classification/secnet_full_ha_aug_s42_v2_ai2/*seed42.npz 2>/dev/null
