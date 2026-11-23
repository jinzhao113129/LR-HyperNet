#!/usr/bin/env bash
# 联网端下载官方原始包，接收方离线解压并生成，不依赖HPC账号。
set -euo pipefail
ARCHIVE=${1:?参数1: 官方Caltech101.zip路径}
OUT=${2:?参数2: 全新H5输出目录}
ENV=${3:-$HOME/envs/hgate}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
test -f "$ARCHIVE"
test ! -e "$OUT"
EXTRACT="${OUT}_raw_extracted"
test ! -e "$EXTRACT"
mkdir -p "$(dirname "$OUT")"
"$ENV/bin/python" "$ROOT/verify_raw.py" "$ARCHIVE" > "${OUT}_archive_check.json"
mkdir "$EXTRACT"
"$ENV/bin/python" -m zipfile -e "$ARCHIVE" "$EXTRACT"
if [ ! -d "$EXTRACT/Caltech101" ]; then
  echo '未发现预期Caltech101类别根目录，请回传解压目录树，不猜路径重跑。'
  exit 2
fi
"$ENV/bin/python" -u "$ROOT/prepare_data.py" --raw-root "$EXTRACT/Caltech101" --out "$OUT" --seed 1234 --num-points 10240 2>&1 | tee "${OUT}_prepare.log"
"$ENV/bin/python" "$ROOT/preflight.py" --data "$OUT" > "${OUT}_data_check.json"
echo "DATA_READY: $OUT；原始目录和日志保留，不自动删除。"
