#!/bin/bash
# 检查各环境 torch 版本与 V100 兼容性
for env in base 100+ ai-coding-benchmark bhli ELLE eventprobe; do
  echo "=== $env ==="
  PY=/home/jnguan/.miniconda/envs/$env/bin/python
  [ -x "$PY" ] || { echo "no python"; continue; }
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1
done
