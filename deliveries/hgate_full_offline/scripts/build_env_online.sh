#!/usr/bin/env bash
# 仅在联网Linux构建机执行；conda须已安装，构建机glibc不高于目标机。
set -euo pipefail
OUT=${1:?用法: bash build_env_online.sh 全新输出目录}
MACHINE=${2:?第二参数必须是目标机machine_info.txt}
test -f "$MACHINE"
test ! -e "$OUT"
test "$(uname -m)" = x86_64
command -v conda >/dev/null
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cp "$MACHINE" "$OUT/target_machine_info.txt"
echo '人工核对：目标Linux x86_64、glibc兼容、驱动支持CUDA12.1。未知则停止，勿猜版本。'
test "${TARGET_COMPATIBILITY_CONFIRMED:-no}" = yes
conda create -y -p "$OUT/build_env" python=3.10.14 pip=24.0
PY="$OUT/build_env/bin/python"
"$PY" -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install numpy==1.26.4 h5py==3.11.0 conda-pack==0.8.1
"$PY" -m pip check
"$PY" -m pip freeze > "$OUT/requirements.lock"
mkdir "$OUT/wheelhouse"
"$PY" -m pip download --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu121 -r "$OUT/requirements.lock" -d "$OUT/wheelhouse"
"$OUT/build_env/bin/conda-pack" -p "$OUT/build_env" -o "$OUT/hgate_py310_cu121.tar.gz"
conda list -p "$OUT/build_env" --explicit > "$OUT/conda-explicit.txt"
"$PY" -c 'import torch,numpy,h5py; print(torch.__version__,torch.version.cuda,numpy.__version__,h5py.__version__)' > "$OUT/build_versions.txt"
sha256sum "$OUT/hgate_py310_cu121.tar.gz" "$OUT/wheelhouse/"* "$OUT/requirements.lock" > "$OUT/SHA256SUMS.absolute"
echo '构建完成；运行seal_transfer.py为最终搬运目录生成相对路径校验清单。'
