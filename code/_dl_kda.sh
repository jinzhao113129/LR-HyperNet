#!/bin/bash
# 在 HPC 登录节点下载 KDA 谱系论文（本地网络访问 arxiv 超时，HPC 可访问）
cd /home/jnguan/JiangYouxin/ZhaoJin
mkdir -p _kda_dl
curl -sL -m 300 -o _kda_dl/KimiLinear_2025.pdf https://arxiv.org/pdf/2510.26692
echo "KimiLinear exit=$?"
curl -sL -m 600 -o _kda_dl/KimiK3_2026.pdf https://arxiv.org/pdf/2607.24653
echo "KimiK3 exit=$?"
curl -sL -m 300 -o _kda_dl/GatedDeltaNet_2024.pdf https://arxiv.org/pdf/2412.06464
echo "GatedDeltaNet exit=$?"
curl -sL -m 300 -o _kda_dl/DeltaNet_2024.pdf https://arxiv.org/pdf/2406.06484
echo "DeltaNet exit=$?"
curl -sL -m 300 -o _kda_dl/LinearAttention_2020.pdf https://arxiv.org/pdf/2006.16236
echo "LinearAttention exit=$?"
echo "--- files:"
ls -la _kda_dl/
echo "--- %PDF check:"
for f in _kda_dl/*.pdf; do
  h=$(head -c 5 "$f" | tr -d '\0')
  echo "$f : $h"
done
