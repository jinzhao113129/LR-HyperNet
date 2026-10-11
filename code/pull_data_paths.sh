#!/bin/bash
# 提取每个实验实际使用的 data_path / epoch / batch（只读）
cd /home/jnguan/JiangYouxin/ZhaoJin

echo '=== HPC 上所有数据目录 ==='
ls -d */ 2>/dev/null | grep -iE 'ncal|cifar|dvs|nmnist|data|official' | sort

echo
echo '=== lr_hypernet 系列：实验名 | data_path | epoch | batch ==='
for f in log/lr_hypernet/*/logs/lr_hypernet.txt; do
  [ -f "$f" ] || continue
  name=$(basename $(dirname $(dirname "$f")))
  dp=$(grep -oE "data_path='[^']*'" "$f" 2>/dev/null | head -1 | sed "s/data_path='//;s/'$//")
  [ -z "$dp" ] && dp=$(grep -oE 'data_path="[^"]*"' "$f" 2>/dev/null | head -1 | sed 's/data_path="//;s/"$//')
  ep=$(grep -oE "epoch=[0-9]+" "$f" 2>/dev/null | head -1)
  bs=$(grep -oE "batch_size=[0-9]+" "$f" 2>/dev/null | head -1)
  echo "$name | $dp | $ep | $bs"
done

echo
echo '=== classification 基线：实验名 | data_path | epoch | batch ==='
for f in log/classification/*/logs/*.txt; do
  [ -f "$f" ] || continue
  name=$(basename $(dirname $(dirname "$f")))
  dp=$(grep -oE "data_path='[^']*'" "$f" 2>/dev/null | head -1 | sed "s/data_path='//;s/'$//")
  [ -z "$dp" ] && dp=$(grep -oE 'data_path="[^"]*"' "$f" 2>/dev/null | head -1 | sed 's/data_path="//;s/"$//')
  ep=$(grep -oE "epoch=[0-9]+" "$f" 2>/dev/null | head -1)
  bs=$(grep -oE "batch_size=[0-9]+" "$f" 2>/dev/null | head -1)
  echo "$name | $dp | $ep | $bs"
done
