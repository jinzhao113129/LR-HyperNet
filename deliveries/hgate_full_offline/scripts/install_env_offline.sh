#!/usr/bin/env bash
# 只在全新目录解包，不联网、不改系统Python、不安装/升级驱动。
set -euo pipefail
BUNDLE=${1:?参数1: 环境离线包目录}
ENV=${2:?参数2: 全新环境绝对路径}
test ! -e "$ENV"
case "$ENV" in /*) ;; *) echo '环境目录必须绝对路径'; exit 2;; esac
test -f "$BUNDLE/hgate_py310_cu121.tar.gz"
mkdir -p "$ENV"
tar -xzf "$BUNDLE/hgate_py310_cu121.tar.gz" -C "$ENV"
"$ENV/bin/python" "$ENV/bin/conda-unpack"
"$ENV/bin/python" -m pip install --no-index --find-links "$BUNDLE/wheelhouse" -r "$BUNDLE/requirements.lock"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -c 'import torch,numpy,h5py; assert torch.cuda.is_available(); print("ENV_OK",torch.__version__,torch.version.cuda,"GPUS",torch.cuda.device_count())'
