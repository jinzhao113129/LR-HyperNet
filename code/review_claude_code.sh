#!/bin/bash
cd /home/jnguan/JiangYouxin/ZhaoJin
echo '=== 1. EuclideanRemoteAggregator 实现 ==='
sed -n '440,520p' models/local_remote_grouper.py
echo
echo '=== 2. LocalRemoteGrouper init 分支 ==='
grep -n 'hyperbolic_type\|euclidean\|Euclidean\|Hyperbolic' models/local_remote_grouper.py | head -20
echo
echo '=== 3. train_lr_hyper.py 改动 ==='
grep -n 'euclidean\|sanity_5class\|temperature\|num_category' train_lr_hyper.py | head -30
echo
echo '=== 4. v3.1 修复版最新进度 ==='
tail -6 log/lr_hypernet/lr_hyper_v31_fix_v100/logs/lr_hypernet.txt
echo
echo '=== 5. 作业状态 ==='
squeue -u jnguan -o '%.10i %.22j %.8T %.10M %.12L %.6R' 2>/dev/null
