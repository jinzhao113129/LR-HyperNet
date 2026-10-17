#!/bin/bash
# ============================================================================
# 3090 平台（学长租约 124，gpu-server-3）一键快测脚本 2026-10-18
#
# 学长规矩（逐字执行）：
#   1. 只用租约 124 的 3090，不动别的 GPU
#   2. 文件不 SCP/拖拽——代码走 git clone，数据走 GitHub Release 下载
#   3. 不做任何 delete 类操作（本脚本零 rm；目录存在即跳过）
#   4. 卡 24h 跑满——6 条链串行执行
#   5. 所有产物在 ~/zhaojin/ 下
#
# 用法（在 gpu-server-3 的 code-server 终端，~/zhaojin/pi2_code 已 clone）：
#   cd ~/zhaojin/pi2_code/scripts
#   bash run_3090_setup_and_run.sh setup    # 第一次：组装布局+装环境（~20分钟）
#   bash run_3090_setup_and_run.sh chain    # 跑 6 条快测链（串行，可 Ctrl-C 后重跑续）
#
# 组装后的布局（HPC 侧 ~/zhaojin/pi2 同构，两边可对照）：
#   ~/zhaojin/pi2/
#     ├── tools/train_frequency_ncaltech.py   ← fork_ai2/scripts/local_edit/
#     ├── models/{FECNet,modules,local_remote_grouper}.py
#     ├── provider_data.py augment_data.py    ← code/
#     ├── data/{train.h5 test.h5}             ← GitHub Release data-baidu-small
#     └── log/classification/<name>/          ← 训练产物
# ============================================================================
set -u  # 不用 -e：链失败不中断后续链（每条链独立判成败）

REPO=~/zhaojin/pi2_code
WORK=~/zhaojin/pi2
PY=~/zhaojin/venv/bin/python
DATA_SHA256=085472b69c3567d48662151b0c46cf168eda984dfdada9dc217056fe23c9428b
DATA_URL=https://github.com/jinzhao113129/LR-HyperNet/releases/download/data-baidu-small/baidu_small.tar.gz

# ---- 6 条快测链定义（口径：test acc 单窗对照官方 Table 9 的 60.60）----
# 协议统一：baidu 3596/934 + augment×3 + EMA 0.999 + patience 20 + epoch 300
# 每条链 = 同配方 + 单组件差异。组件说明见 run_pi_ha_baidu_ai2.sh 头注。
chain_flags() {
  case "$1" in
    base) echo "" ;;
    ha)   echo "--hyperbolic_attention" ;;
    hsel) echo "--hyperbolic_selector --hsel_scaled_coords" ;;
    spike) echo "--spike_time_enc" ;;
    rp1)  echo "--rp1_selector" ;;
    hmlp) echo "--hyperbolic_attention --hyperbolic_mlp" ;;
    *) echo "UNKNOWN_CHAIN"; exit 2 ;;
  esac
}
CHAIN_ORDER="base ha hsel spike rp1 hmlp"

# ============================================================================
# setup：组装布局 + 环境安装（幂等，重复执行安全）
# ============================================================================
do_setup() {
  echo "=== [1/4] 组装 ~/zhaojin/pi2 布局（无 delete，目录存在即跳过） ==="
  mkdir -p "$WORK/tools" "$WORK/models" "$WORK/code/analysis" "$WORK/data" "$WORK/log"

  # 代码：从 clone 的仓库复制（本地复制不违反禁 SCP——不经过本机）
  for f in train_frequency_ncaltech.py extract_preds.py; do
    [ -f "$WORK/tools/$f" ] || cp "$REPO/fork_ai2/scripts/local_edit/$f" "$WORK/tools/"
  done
  for f in FECNet.py modules.py local_remote_grouper.py; do
    [ -f "$WORK/models/$f" ] || cp "$REPO/fork_ai2/scripts/local_edit/$f" "$WORK/models/"
  done
  for f in provider_data.py augment_data.py; do
    [ -f "$WORK/$f" ] || cp "$REPO/code/$f" "$WORK/"
  done
  [ -f "$WORK/code/analysis/mcnemar_test.py" ] || cp "$REPO/code/analysis/mcnemar_test.py" "$WORK/code/analysis/"

  echo "=== [2/4] 数据下载（GitHub Release，release 通道 = 学长认可） ==="
  if [ -f "$WORK/data/train.h5" ] && [ -f "$WORK/data/test.h5" ]; then
    echo "data 已在，跳过下载"
  else
    cd "$WORK/data"
    # 三次重试（GitHub TLS 瞬断对策）
    for i in 1 2 3; do
      curl -sSL -m 3600 -o baidu_small.tar.gz "$DATA_URL" && break
      echo "下载失败（第 $i 次），5 秒后重试..."; sleep 5
    done
    echo "SHA256 校验..."
    ACTUAL=$(sha256sum baidu_small.tar.gz | cut -d' ' -f1)
    if [ "$ACTUAL" != "$DATA_SHA256" ]; then
      echo "!! SHA256 不匹配：$ACTUAL（期望 $DATA_SHA256）"
      echo "!! 数据包损坏，请重跑 setup（下载文件保留，可手动重下）"
      return 1
    fi
    tar xzf baidu_small.tar.gz
    mv ncaltech101_official_baidu/train.h5 ncaltech101_official_baidu/test.h5 .
    echo "数据就位："
    ls -la train.h5 test.h5
  fi

  echo "=== [3/4] Python 环境（venv，不动系统 conda） ==="
  if [ -x "$PY" ]; then
    echo "venv 已存在，跳过"
  else
    python3 -m venv ~/zhaojin/venv
    "$PY" -m pip install --upgrade pip
    # torch 2.3.1+cu121（3090 = sm_86，cu121 支持）+ 依赖（import 清单实测）
    "$PY" -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    "$PY" -m pip install "numpy<2" h5py scikit-learn tensorboard
  fi

  echo "=== [4/4] 编译自检 ==="
  cd "$WORK"
  "$PY" -m py_compile tools/train_frequency_ncaltech.py models/FECNet.py \
    models/modules.py models/local_remote_grouper.py provider_data.py augment_data.py \
    && echo COMPILE_OK

  echo "=== setup 完成 ==="
  echo "下一步：bash run_3090_setup_and_run.sh chain"
}

# ============================================================================
# chain：跑 6 条快测链（串行；断点续跑 = 重跑本命令，已完成的链自动跳过）
# ============================================================================
do_chain() {
  echo "=== 检查环境 ==="
  [ -x "$PY" ] || { echo "venv 不存在，先跑 setup"; return 1; }
  [ -f "$WORK/data/train.h5" ] || { echo "数据不在，先跑 setup"; return 1; }

  cd "$WORK"
  export PYTHONPATH="$WORK:$WORK/models"
  mkdir -p ~/zhaojin/logs_run

  for name in $CHAIN_ORDER; do
    LOG=~/zhaojin/logs_run/${name}.log
    MARK=~/zhaojin/logs_run/${name}.done
    if [ -f "$MARK" ]; then
      echo "--- [$name] 已完成（$(cat $MARK)），跳过 ---"
      continue
    fi
    echo "=== [$(date '+%m-%d %H:%M')] 链 $name 开始：$(chain_flags $name) ==="
    # nohup 不用：code-server 终端断开也不杀（code-server 的 shell 是
    # persistent 的），但保险起见用 setsid 脱离会话
    setsid "$PY" tools/train_frequency_ncaltech.py \
      --data_path ./data/ \
      --num_point 10240 \
      --batch_size 16 \
      --grad_accum 2 \
      --epoch 300 \
      --patience 20 \
      --early_stop_metric acc \
      --learning_rate 0.001 \
      --optimizer AdamW \
      --decay_rate 0.0001 \
      --num_category 101 \
      --log_dir secnet_pi2_${name}_baidu \
      --log_path ./log/ --log_name /secnet_pi2_${name}_baidu \
      --augment 3 \
      --ema 0.999 \
      --seed 42 \
      $(chain_flags $name) \
      > "$LOG" 2>&1
    RC=$?
    echo "链 $name 退出码 $RC"
    if [ $RC -eq 0 ]; then
      # 提取 best acc 写入 mark（事后汇总用）
      BEST=$(grep -a "best test acc" "$LOG" | tail -1 | grep -aoE "[0-9]+\.[0-9]+%" | tail -1)
      echo "${BEST:-finished} @ $(date '+%m-%d %H:%M')" > "$MARK"
    else
      echo "链 $name 失败（RC=$RC）——看 $LOG 排查后重跑（自动 resume）"
    fi
  done
  echo "=== 全部链执行完毕 ==="
  cat ~/zhaojin/logs_run/*.done 2>/dev/null
}

case "$1" in
  setup) do_setup ;;
  chain) do_chain ;;
  *) echo "用法：bash run_3090_setup_and_run.sh {setup|chain}"; exit 1 ;;
esac
