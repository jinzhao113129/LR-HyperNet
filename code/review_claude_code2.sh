#!/bin/bash
cd /home/jnguan/JiangYouxin/ZhaoJin
echo '=== LocalRemoteGrouper.__init__ 分支（关键） ==='
grep -n 'class LocalRemoteGrouper' models/local_remote_grouper.py
sed -n '/class LocalRemoteGrouper/,/def forward/p' models/local_remote_grouper.py | head -60
echo
echo '=== verify 脚本 ==='
head -50 verify_euclidean_sanity.py 2>/dev/null || echo "文件不存在"
echo
echo '=== 跑 verify 脚本（CPU） ==='
/home/jnguan/.miniconda/envs/bhli/bin/python verify_euclidean_sanity.py 2>&1 | tail -20
echo
echo '=== 根目录陈旧文件检查 ==='
ls -la local_remote_grouper.py 2>/dev/null && echo "存在！" || echo "不存在"
grep -c 'norm_max' local_remote_grouper.py 2>/dev/null || echo "0"
grep -c 'norm_max' models/local_remote_grouper.py 2>/dev/null
