#!/usr/bin/env bash
# 两条固定几何选点按顺序训练；已有 checkpoint 自动跳过。
set -euo pipefail

WORK="$HOME/zhaojin/pi2"
PY="$HOME/zhaojin/venv/bin/python"
LOGS="$HOME/zhaojin/logs_run"
mkdir -p "$LOGS"
cd "$WORK"
export PYTHONPATH="$WORK:$WORK/models:$WORK/tools"

if pgrep -f '[t]ools/train_frequency_ncaltech.py' >/dev/null; then
  echo '已有训练进程在跑，sweep 未启动。' >&2
  exit 2
fi

for treatment in cosine_mix cosine_diverse; do
  name="secnet_pi2_g16_${treatment}_b4_baidu"
  checkpoint="log/classification/$name/checkpoints/last_checkpoint.pth"
  if [[ -f "$checkpoint" ]]; then
    echo "[$treatment] checkpoint 已存在，跳过: $checkpoint"
    continue
  fi
  echo "[$treatment] $(date '+%F %T') 开始"
  "$PY" tools/train_frequency_ncaltech.py \
    --data_path ./data/ --num_point 10240 --batch_size 4 --grad_accum 8 \
    --epoch 300 --patience 20 --early_stop_metric acc \
    --learning_rate 0.001 --optimizer AdamW --decay_rate 0.0001 \
    --num_category 101 --log_dir "$name" --log_path ./log/ \
    --log_name "/$name" --augment 3 --ema 0.999 --seed 42 \
    --remote_g16 --g16_treatment "$treatment" \
    --selector_novelty_weight 0.5 --learned_M 128 --cs_chunk 64 \
    > "$LOGS/g16_${treatment}_b4.log" 2>&1
  echo "[$treatment] $(date '+%F %T') 完成"
done
