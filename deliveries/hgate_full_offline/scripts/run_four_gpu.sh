#!/usr/bin/env bash
# 前台运行，后台运行见README；仅显式使用用户提供的四张卡。
set -euo pipefail
MODE=${1:?用法: run_four_gpu.sh benchmark|train|resume 数据目录 输出目录 [环境目录]}
DATA=${2:?需要数据目录}
OUT=${3:?需要输出目录}
ENV=${4:-$HOME/envs/hgate}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONHASHSEED=42
EXTRA=()
case "$MODE" in
  benchmark) EXTRA=(--mode benchmark --steps 12);;
  train) ;;
  resume) EXTRA=(--resume "$OUT/last_checkpoint.pth");;
  *) echo '模式只能benchmark/train/resume'; exit 2;;
esac
exec "$ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 "$ROOT/train_ddp.py" --data "$DATA" --out "$OUT" "${EXTRA[@]}"
