#!/bin/bash
# 链式接力训练 v5（Codex 2026-10-15 复审修复版）
# 相对 v4 的修复：
# 1. 加 START_SEG 参数——支持"只提交 p2-p6"接续已运行的 p1（v4 会重复提交 p1，
#    与运行中作业并发读写同一 checkpoint，硬阻断）。
# 2. 段边界从实际 checkpoint 对齐——p2 起点由调用方传入 ACTUAL_EP（p1 结束后
#    从 last_checkpoint 读到的实际 epoch），不假设 p1 恰好跑满 STOP_PER_SEG。
# 用法: chain_train.sh <log_dir> <作业名前缀> <每段epoch数> [START_SEG] [ACTUAL_EP] [组件flag...]
# 例: 全新链: chain_train.sh secnet_full_hsel_s42_ai2 sn_fu_hsel 28 1 0 --hyperbolic_selector
# 例: 接续链: chain_train.sh secnet_full_hsel_s42_ai2 sn_fu_hsel 28 2 36 --hyperbolic_selector
#   （p1 TIMEOUT 后：START_SEG=2, ACTUAL_EP=p1 实际完成的 epoch 数）
set -e

EXP=$1; NAME=$2; STOP_PER_SEG=$3; START_SEG=${4:-1}; ACTUAL_EP=${5:-0}
if [ -z "$EXP" ] || [ -z "$NAME" ] || [ -z "$STOP_PER_SEG" ]; then
    echo "用法: $0 <log_dir> <作业名前缀> <每段epoch数> [起始段号=1] [实际已完成epoch=0] [组件flag...]" >&2
    echo "错误: 前 3 参数必须非空（收到: EXP='$EXP' NAME='$NAME' STOP='$STOP_PER_SEG'）" >&2
    exit 1
fi
if ! [[ "$STOP_PER_SEG" =~ ^[0-9]+$ ]] || [ "$STOP_PER_SEG" -le 0 ]; then
    echo "错误: 每段epoch数须为正整数，收到 '$STOP_PER_SEG'" >&2; exit 1
fi
if ! [[ "$START_SEG" =~ ^[0-9]+$ ]] || [ "$START_SEG" -lt 1 ]; then
    echo "错误: 起始段号须 ≥1，收到 '$START_SEG'" >&2; exit 1
fi
if ! [[ "$ACTUAL_EP" =~ ^[0-9]+$ ]] || [ "$ACTUAL_EP" -lt 0 ]; then
    echo "错误: 实际epoch须 ≥0，收到 '$ACTUAL_EP'" >&2; exit 1
fi
if [[ "$EXP" != *_ai2 ]]; then
    echo "警告: log_dir '$EXP' 不带 _ai2 后缀——继续，但请确认" >&2
fi
if [ "$START_SEG" -eq 1 ] && [ "$ACTUAL_EP" -gt 0 ]; then
    echo "错误: START_SEG=1（全新链）时 ACTUAL_EP 必须为 0" >&2; exit 1
fi
shift $(( $# > 5 ? 5 : $# ))
COMPONENT_FLAGS="$@"

TOTAL_EPOCH=150
# 段边界：seg i 的 stop = ACTUAL_EP + (i - START_SEG + 1) * STOP_PER_SEG
# 首段（接续时）从 ACTUAL_EP 起算；末段封顶 TOTAL_EPOCH
SEGS=$(( (TOTAL_EPOCH - ACTUAL_EP + STOP_PER_SEG - 1) / STOP_PER_SEG + START_SEG - 1 ))
if [ "$STOP_PER_SEG" -gt 34 ]; then
    echo "警告: 每段 $STOP_PER_SEG epoch 可能超 12h（A800 实测 19.4min/ep）" >&2
fi
echo "链计划: log_dir=$EXP 前缀=$NAME"
echo "  从段 $START_SEG 到 $SEGS；已完成 $ACTUAL_EP / $TOTAL_EPOCH epoch"

cd /home/jnguan/JiangYouxin/ZhaoJin
PREV_JOBID=""
for i in $(seq $START_SEG $SEGS); do
    OFFSET=$(( i - START_SEG ))
    STOP_EP=$(( ACTUAL_EP + (OFFSET + 1) * STOP_PER_SEG ))
    if [ $STOP_EP -gt $TOTAL_EPOCH ]; then STOP_EP=$TOTAL_EPOCH; fi
    SB=chain_${NAME}_p${i}.sbatch
    cat > $SB << EOF
#!/bin/bash
#SBATCH -J ${NAME}_p${i}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/${NAME}_ai2_%j.out
#SBATCH -e eval_results/${NAME}_ai2_%j.err
set -e
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:\$PYTHONPATH
# 段 ${i}/${SEGS}: 从 last_checkpoint 续训（配置一致性校验会挡错配）
# 本段跑到 epoch ${STOP_EP} 正常退出（--stop_epoch，Codex 案 2）
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \\
    --data_path ./ncaltech101_official_gen/ \\
    --num_point 10240 \\
    --batch_size 32 \\
    --epoch ${TOTAL_EPOCH} \\
    --stop_epoch ${STOP_EP} \\
    --learning_rate 0.001 \\
    --optimizer AdamW \\
    --decay_rate 0.0001 \\
    --num_category 101 \\
    --log_dir ${EXP} \\
    --log_name /${EXP} \\
    --grad_accum 2 \\
    --seed 42 ${COMPONENT_FLAGS}
EOF
    # 接续模式（START_SEG>1）首段不带依赖（p1 已离队，checkpoint 手动对齐）
    if [ -n "$PREV_JOBID" ]; then
        OUT=$(sbatch --no-requeue --dependency=afterok:${PREV_JOBID} "$SB" 2>&1)
    else
        OUT=$(sbatch --no-requeue "$SB" 2>&1)
    fi
    JOBID=$(echo "$OUT" | grep -oE '[0-9]+$')
    if [ -z "$JOBID" ]; then echo "段 $i 提交失败: $OUT" >&2; exit 1; fi
    echo "段 ${i}/${SEGS} (epoch ${ACTUAL_EP}→${STOP_EP}) -> 作业 ${JOBID}"
    PREV_JOBID=$JOBID
done
echo "=== 链提交完成: ${NAME} 段 ${START_SEG}-${SEGS}, 末段 ${PREV_JOBID} ==="
