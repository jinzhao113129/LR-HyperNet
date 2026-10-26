#!/bin/bash
# ============================================================================
# 剩余链自动串行续跑（2026-10-26）
# 顺序：fca → polar → g16_cosine（cosine_near，找回学长 RP1 的 7pp，最重要）
# 目的：平台拆机器后「租用卡半小时无进程自动降级 CPU」，串行脚本保证一条跑完
#       立刻接下一条、中间不空转。
# 断点续跑：每条链跑完写 ~/zhaojin/logs_run/<name>.done，重跑本脚本自动跳过已完成的。
#
# 用法（SSH 断也不影响，nohup 后台跑）：
#   cd ~/zhaojin/pi2
#   nohup bash scripts/run_remaining_chain.sh > ~/zhaojin/logs_run/auto_chain.log 2>&1 &
#
# 注意：本脚本内训练用前台阻塞（不用 setsid），训练早停退出后自动接下一条；
#      脚本本身用 nohup 起，所以 SSH 断不影响串行推进。
# ============================================================================
set -u
WORK=~/zhaojin/pi2
PY=~/zhaojin/venv/bin/python
MARK_DIR=~/zhaojin/logs_run
mkdir -p "$MARK_DIR"

cd "$WORK" || { echo "无 $WORK"; exit 1; }
export PYTHONPATH="$WORK:$WORK/models"

# 链配置：名字 | 额外 flags | batch | accum
# fca/polar 是组件链（batch16/accum2）；g16_cosine 是 G16 链（batch8/accum4，双曲距离矩阵显存贵）
run_chain() {
  local name="$1" flags="$2" bs="$3" ga="$4"
  local mark="$MARK_DIR/${name}.done"
  if [ -f "$mark" ]; then
    echo "[$name] 已完成（$(cat "$mark")），跳过"
    return 0
  fi
  echo "=== $(date '+%m-%d %H:%M') 起 [$name]（batch=$bs accum=$ga）flags=$flags ==="
  "$PY" tools/train_frequency_ncaltech.py \
    --data_path ./data/ --num_point 10240 --batch_size "$bs" --grad_accum "$ga" \
    --epoch 300 --patience 20 --early_stop_metric acc \
    --learning_rate 0.001 --optimizer AdamW --decay_rate 0.0001 \
    --num_category 101 --log_dir "secnet_pi2_${name}_baidu" \
    --log_path ./log/ --log_name "/secnet_pi2_${name}_baidu" \
    --augment 3 --ema 0.999 --seed 42 $flags \
    > "$MARK_DIR/${name}.log" 2>&1
  local rc=$?
  echo "[$name] 退出码 $rc @ $(date '+%m-%d %H:%M')"
  # 提取 best acc 写入 done 标记（事后汇总用）
  local best
  best=$(grep -a "best test acc" "$MARK_DIR/${name}.log" 2>/dev/null | tail -1)
  echo "$rc @ $(date '+%m-%d %H:%M') ${best:-}" > "$mark"
}

# 串行执行（前台阻塞，一条早停完自动接下一条）
run_chain fca "--fca_gate" 16 2
run_chain polar "--polar_feature" 16 2
run_chain g16_cosine "--remote_g16 --g16_treatment cosine_near" 8 4

echo "=== 全部链执行完毕 ==="
cat "$MARK_DIR"/*.done 2>/dev/null
