#!/bin/bash
echo "=== v3 smoke 167733 结果 ==="
sacct -j 167733 --format=JobID,JobName,State,Elapsed,NodeList -P 2>/dev/null
echo ""
echo "=== v3 smoke 日志末尾 ==="
ls -lt /home/jnguan/JiangYouxin/ZhaoJin/log/lr_hypernet/lr_hyper_v3_smoke_h100/logs/ 2>/dev/null | head -5
tail -30 /home/jnguan/JiangYouxin/ZhaoJin/log/lr_hypernet/lr_hyper_v3_smoke_h100/logs/*.txt 2>/dev/null | tail -40
echo ""
echo "=== 167959 日志目录 ==="
ls -lt /home/jnguan/JiangYouxin/ZhaoJin/log/classification/secnet*/logs/ 2>/dev/null | head -10
echo ""
echo "=== 167959 最新日志 ==="
find /home/jnguan/JiangYouxin/ZhaoJin/log -name "*.txt" -newer /tmp -mmin -60 2>/dev/null | head -5
