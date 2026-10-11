#!/bin/bash
cd /home/jnguan/JiangYouxin/ZhaoJin
echo '=== hfecnet 目录 ==='
ls -d log/hfecnet/*/ 2>/dev/null
echo
for f in log/hfecnet/*/logs/*.txt; do
  [ -f "$f" ] || continue
  n=$(grep -c 'INFO - Test Accuracy: ' "$f" 2>/dev/null)
  lastt=$(grep 'INFO - Test Accuracy: ' "$f" 2>/dev/null | tail -1 | sed 's/.*INFO - Test Accuracy: //;s/ *$//')
  lasttr=$(grep 'INFO - Train Accuracy: ' "$f" 2>/dev/null | tail -1 | sed 's/.*INFO - Train Accuracy: //;s/ *$//')
  dp=$(grep -oE "data_path='[^']*'" "$f" 2>/dev/null | head -1 | sed "s/data_path='//;s/'$//")
  echo "[${n}ep] $(basename $(dirname $(dirname "$f"))) | data=$dp | Train=$lasttr | Test=$lastt"
done
