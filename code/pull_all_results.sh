#!/bin/bash
# 汇总所有实验最终分数 v2（只读，精确提取非 seq 行）
cd /home/jnguan/JiangYouxin/ZhaoJin

echo '======== lr_hypernet 系列 ========'
for f in log/lr_hypernet/*/logs/lr_hypernet.txt; do
  [ -f "$f" ] || continue
  n=$(grep -c 'INFO - Test Accuracy: ' "$f")
  lastt=$(grep 'INFO - Test Accuracy: ' "$f" | tail -1 | sed 's/.*INFO - Test Accuracy: //;s/ *$//')
  lasttr=$(grep 'INFO - Train Accuracy: ' "$f" | tail -1 | sed 's/.*INFO - Train Accuracy: //;s/ *$//')
  echo "[${n}ep] $(basename $(dirname $(dirname "$f"))) | Train: $lasttr | Test: $lastt"
done

echo
echo '======== classification 基线 ========'
for f in log/classification/*/logs/*.txt; do
  [ -f "$f" ] || continue
  n=$(grep -c 'INFO - Test Accuracy: ' "$f")
  lastt=$(grep 'INFO - Test Accuracy: ' "$f" | tail -1 | sed 's/.*INFO - Test Accuracy: //;s/ *$//')
  lasttr=$(grep 'INFO - Train Accuracy: ' "$f" | tail -1 | sed 's/.*INFO - Train Accuracy: //;s/ *$//')
  echo "[${n}ep] $(basename $(dirname $(dirname "$f"))) | Train: $lasttr | Test: $lastt"
done
