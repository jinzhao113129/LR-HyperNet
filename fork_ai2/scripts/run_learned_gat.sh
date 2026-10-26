#!/bin/bash
# ============================================================================
# learned_v2（GAT 式交互 scorer + 4 路信号）单链快测（2026-10-26）
# 与 run_remaining_chain.sh 串行：等 fca→polar→g16_cosine 全跑完后自动接上，
# 避免平台「租用卡半小时无进程降级 CPU」。
#
# 用法（SSH 断也不影响）：
#   nohup bash -c 'while pgrep -f "run_remaining_chain[.]sh" >/dev/null; do sleep 60; done; \
#     bash ~/zhaojin/pi2/run_learned_gat.sh' \
#     > ~/zhaojin/logs_run/learned_gat_watch.log 2>&1 &
#
# 注：pgrep 模式写成 run_remaining_chain[.]sh，避免匹配到 watcher 自身命令行。
# 日志目录用 g16_learned_gat，避开旧 learned_v2 日志（那是 LapSum 的第一版）。
# ============================================================================
set -u
WORK=~/zhaojin/pi2
PY=~/zhaojin/venv/bin/python
MARK_DIR=~/zhaojin/logs_run
mkdir -p "$MARK_DIR"
cd "$WORK" || { echo "无 $WORK"; exit 1; }
export PYTHONPATH="$WORK:$WORK/models"

NAME=g16_learned_gat
echo "=== $(date '+%m-%d %H:%M') 起 [$NAME]（batch=8 accum=4）flags=--remote_g16 --g16_treatment learned_v2 ===" \
  >> "$MARK_DIR/auto_chain.log"
"$PY" tools/train_frequency_ncaltech.py \
  --data_path ./data/ --num_point 10240 --batch_size 8 --grad_accum 4 \
  --epoch 300 --patience 20 --early_stop_metric acc \
  --learning_rate 0.001 --optimizer AdamW --decay_rate 0.0001 \
  --num_category 101 --log_dir "secnet_pi2_${NAME}_baidu" \
  --log_path ./log/ --log_name "/secnet_pi2_${NAME}_baidu" \
  --augment 3 --ema 0.999 --seed 42 \
  --remote_g16 --g16_treatment learned_v2 \
  > "$MARK_DIR/${NAME}.log" 2>&1
rc=$?
echo "[$NAME] 退出码 $rc @ $(date '+%m-%d %H:%M')" >> "$MARK_DIR/auto_chain.log"
best=$(grep -a "best test acc" "$MARK_DIR/${NAME}.log" 2>/dev/null | tail -1)
echo "$rc @ $(date '+%m-%d %H:%M') ${best:-}" > "$MARK_DIR/${NAME}.done"
