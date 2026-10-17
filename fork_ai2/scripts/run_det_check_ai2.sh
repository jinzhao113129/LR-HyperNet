#!/bin/bash
#SBATCH -J det_check_ai2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o eval_results/det_check_ai2_%j.out
#SBATCH -e eval_results/det_check_ai2_%j.err

cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH

# 2026-10-14 ai2: 确定性验证——同一 checkpoint 同 eval_seed 连跑两次推理，
# 检验"固定seed推理可复现"是否成立（发现 rp1_fix 两次结果差 1.18pp）。
# 两次结果分别存 run1/run2 后缀（不改标准 last_seed42.npz 的内容语义，
# 事后会记录并清理这两个临时文件）。
/home/jnguan/.miniconda/envs/bhli/bin/python - <<'PYEOF'
import sys
sys.argv = ['x']
import tools.extract_preds as ep
import torch, numpy as np, random

device = torch.device('cuda', 0)

def run_once():
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    np.random.seed(42); random.seed(42)
    return ep.extract_one('secnet_rp1_fix_v100', 'last', './ncaltech101_official_baidu/', device, 10240, 101, 8, 42)

# 跑两次，extract_one 内部会写 last_seed42.npz，先备份第一次
import shutil, os
acc1 = run_once()
shutil.copy('log/classification/secnet_rp1_fix_v100/preds/last_seed42.npz',
            '/tmp/det_check_run1.npz')
acc2 = run_once()
shutil.copy('log/classification/secnet_rp1_fix_v100/preds/last_seed42.npz',
            '/tmp/det_check_run2.npz')
d1 = np.load('/tmp/det_check_run1.npz')
d2 = np.load('/tmp/det_check_run2.npz')
same = (d1['y_pred'] == d2['y_pred']).sum()
print('=== DETERMINISM CHECK ===')
print('run1 acc: %.4f  run2 acc: %.4f' % (acc1, acc2))
print('identical predictions: %d / %d' % (same, len(d1['y_pred'])))
print('VERDICT:', 'REPRODUCIBLE' if same == len(d1['y_pred']) else 'NON-DETERMINISTIC — 噪声源存在')
PYEOF
