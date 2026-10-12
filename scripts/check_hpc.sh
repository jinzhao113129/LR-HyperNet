#!/bin/bash
echo "=== 当前作业 ==="
squeue -u jnguan -o "%.10i %.9P %.25j %.8u %.2t %.10M %.6D %R"

echo ""
echo "=== 最近完成的作业（前15条）==="
sacct -u jnguan --starttime=2026-10-09 --format=JobID,JobName,State,Elapsed,NodeList -P | head -20

echo ""
echo "=== GPU 节点状态 ==="
sinfo -N -o "%N %t %G %m" | /bin/grep -E "gpu|idle|mix" | head -30

echo ""
echo "=== 167959 详情 ==="
scontrol show job 167959 | /bin/grep -E "JobName|NodeList|RunTime|TimeLimit|TRES|Partition"
