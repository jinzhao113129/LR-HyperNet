#!/bin/bash
# A100-40GB 版链式接力训练（独立 sbatch 前缀 chainA_，防与 V100 链覆盖）
# 用法: chain_train_a100.sh <log_dir名> <作业名前缀> <段数> [组件flag...]
set -e
EXP=$1      # log_dir 名（如 secnet_fullA_base_s42_ai2）
NAME=$2     # 作业名前缀（如 sn_fuA_base）
SEGS=$3
shift 3
EXTRA_FLAGS="$@"

cd /home/jnguan/JiangYouxin/ZhaoJin
PREV_JOBID=""
for i in $(seq 1 $SEGS); do
    SB=chainA_${NAME}_${i}.sbatch
    cat > $SB << EOF
#!/bin/bash
#SBATCH -J ${NAME}_p${i}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o eval_results/${NAME}_ai2_%j.out
#SBATCH -e eval_results/${NAME}_ai2_%j.err
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:\$PYTHONPATH
# 段 ${i}/${SEGS}（A100-40GB，batch 32）：自动从 last_checkpoint 续训
/home/jnguan/.miniconda/envs/bhli/bin/python train_frequency_ncaltech.py \\
    --data_path ./ncaltech101_official_gen/ \\
    --num_point 10240 \\
    --batch_size 32 \\
    --epoch 150 \\
    --learning_rate 0.001 \\
    --optimizer AdamW \\
    --decay_rate 0.0001 \\
    --num_category 101 \\
    --log_dir ${EXP} \\
    --log_name /${EXP} \\
    --seed 42 ${EXTRA_FLAGS}
EOF
    if [ -n "$PREV_JOBID" ]; then
        OUT=$(sbatch --dependency=afterok:${PREV_JOBID} $SB 2>&1)
    else
        OUT=$(sbatch $SB 2>&1)
    fi
    JOBID=$(echo "$OUT" | grep -oE '[0-9]+$')
    echo "段 ${i}/${SEGS} -> 作业 ${JOBID} (${SB})"
    PREV_JOBID=$JOBID
done
echo "=== A100 链提交完成: ${NAME} 共 ${SEGS} 段 ==="
