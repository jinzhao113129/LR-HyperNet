#!/usr/bin/env bash
# 只读采集，不安装软件、不更改驱动；请回传整个输出。
set -eu
date -u
uname -a
cat /etc/os-release
getconf GNU_LIBC_VERSION || true
command -v python3 || true
python3 --version || true
free -h
df -h . /tmp
lscpu | head -25
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
nvidia-smi topo -m
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
if command -v python3 >/dev/null; then
  python3 -c 'import importlib.util,platform; print("ARCH",platform.machine()); print("TORCH_INSTALLED",bool(importlib.util.find_spec("torch")))'
fi
