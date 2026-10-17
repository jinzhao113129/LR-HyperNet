#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LRHyperNet 专用推理脚本（G16 六处理 / 双曲远程聚合 checkpoint 评测）。

现有 tools/extract_preds.py 只支持 FECNet（train_frequency_ncaltech.py 路径），
不能评 LRHyperNet（train_lr_hyper.py 路径）的 checkpoint——G16 六处理训练的
产物没有评测入口。本脚本补齐：

  - 从 checkpoint 自己的 `args`（save_model 已存 vars(args)）重建 LRHyperNet，
    构造参数与训练逐键一致（g16_treatment/g16_rng_seed/selector_mode/
    hyperbolic_type/use_hyperbolic_selector/use_rp1_selector 等全部恢复）；
  - 固定 eval_seed=42（Codex 要求），每 (log_dir, which, eval_seed) 独立重播种
    （同 extract_preds.py 的修复：FPS torch.randint 依赖 RNG 状态，必须重置）；
  - 推理时 g16_treatment 生效（hyp_far/euc_near 选点路径走真实 treatment 映射），
    但不挂 --g16_profile（sink 不采集，纯前向）。

输出：log/lr_hypernet/<log_dir>/<which>_seed<eval_seed>.npz（无中间 preds/ 层，
直接对齐 mcnemar_test.py 的读取路径）
  keys: y_true, y_pred, acc, log_dir, which, eval_seed, epoch

============================================================================
上传位置（Codex 复审：REPO_ROOT 依赖文件所在目录层级）：
  本文件必须上传到 HPC 的 `tools/extract_preds_lrh.py`（与 tools/extract_preds.py
  同级），这样 REPO_ROOT = dirname(dirname(__file__)) = /home/jnguan/JiangYouxin/
  ZhaoJin，`import provider_data` / `from lr_hypernet import` / 输出路径
  log/lr_hypernet/<log_dir> 才都正确。
  切勿放在 fork_ai2/scripts/（REPO_ROOT 会算成 fork_ai2，import 全错）。
============================================================================
注意：测试窗口数是 official_gen 的 5226（1742 视频 × 3 窗），不是旧
  baidu 协议的 934——后续 McNemar 脚本和结论表按 5226 对齐。

用法：
  python extract_preds_lrh.py --log_dir lr_hyper_g16_hypfar_s42_ai2 --which both
  python extract_preds_lrh.py --log_dirs lr_hyper_g16_hypfar_s42_ai2,lr_hyper_g16_eucnear_s42_ai2 --which last
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))
sys.path.insert(0, os.path.join(REPO_ROOT, "ai2_g16"))

import provider_data  # noqa: E402
from lr_hypernet import LRHyperNet  # noqa: E402

# LRHyperNet 构造参数（与 train_lr_hyper.py main() 的 MODEL LOADING 逐键一致）。
# 注意：g16_rng_seed 由 args.seed 派生（train_lr_hyper.py 里 args.g16_rng_seed
# = args.seed），checkpoint 的 args 里两者都存了；这里用 seed 兜底旧 ckpt。
_LRH_BUILD_KEYS = (
    "remote_g", "curvature_init", "time_enc", "use_global_anchor",
    "anchor_lambda", "kda", "kda_bidirectional", "hyperbolic_type",
    "hyperbolic_classifier", "group_norm", "use_hyperbolic_selector",
    "remote_dropout", "use_rp1_selector", "rp1_lambda", "selector_mode",
)


def build_lrh(saved_args, num_category):
    """按 checkpoint 存的 args 重建 LRHyperNet（与训练构造逐键一致）。"""
    a = saved_args
    return LRHyperNet(
        num_classes=num_category,
        remote_g=int(a.get("remote_g", 16)),
        curvature_init=float(a.get("curvature_init", 0.4)),
        time_enc=bool(a.get("time_enc", False)),
        use_global_anchor=bool(a.get("use_global_anchor", False)),
        anchor_lambda=float(a.get("anchor_lambda", 0.5)),
        kda=bool(a.get("kda", False)),
        kda_bidirectional=bool(a.get("kda_bidirectional", False)),
        hyperbolic_type=str(a.get("hyperbolic_type", "poincare")),
        hyperbolic_classifier=bool(a.get("hyperbolic_classifier", False)),
        use_group_norm=bool(a.get("group_norm", False)),
        use_hyperbolic_selector=bool(a.get("use_hyperbolic_selector", False)),
        remote_dropout=float(a.get("remote_dropout", 0.0)),
        use_rp1_selector=bool(a.get("use_rp1_selector", False)),
        rp1_lambda=float(a.get("rp1_lambda", 1.0)),
        selector_mode=str(a.get("selector_mode", "topk")),
        # G16 六处理分派：g16_treatment 空串 = 原行为（旧 checkpoint 兼容）
        g16_treatment=str(a.get("g16_treatment", "") or ""),
        g16_rng_seed=int(a.get("g16_rng_seed", a.get("seed", 42))),
    )


def load_test_loader(data_path, num_point, batch_size):
    """与 train_lr_hyper.py 的 validate 数据加载逐行一致（shuffle=False）。"""
    test_file = data_path + "test.h5"
    data, label, _mark = provider_data.load_h5_mark(test_file)
    label = np.squeeze(label)
    data = torch.from_numpy(data).reshape(-1, num_point, 4)
    label = torch.from_numpy(label.astype("int64"))
    # 与 train_lr_hyper.py main() 的 test 分支一致：reshape 拆分片段时对齐 label
    if data.shape[0] != label.shape[0]:
        n_seg = data.shape[0]
        rep = n_seg // label.shape[0]
        rem = n_seg % label.shape[0]
        label = torch.cat([label.repeat_interleave(rep), label[-1:].repeat(rem)])
    ds = torch.utils.data.TensorDataset(data, label)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    ), data.shape[0]


def extract_one(log_dir, which, data_path, device, num_point, num_category,
                batch_size, eval_seed):
    # 每 (log_dir, which) 独立重播种（同 extract_preds.py 修复）：FPS 起始点
    # torch.randint 依赖 RNG，必须每轮重置才能让 (log_dir, which, eval_seed)
    # 唯一决定结果。
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)
    ckpt_name = "last_checkpoint.pt" if which == "last" else "best_checkpoint.pt"
    ckpt_path = os.path.join("log", "lr_hypernet", log_dir, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print("[SKIP] %s: no %s" % (log_dir, ckpt_name))
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args")
    if not saved_args:
        print("[SKIP] %s: checkpoint 无 args（无法重建 LRHyperNet）" % log_dir)
        return None

    # Codex P0-2 修复（2026-10-17）：raw/EMA 双输出——net_ema 键存在时
    # 额外评一份（EMA 是并行出口，原 net 输出完全不变）。两份 npz 各自
    # 命名（<which> / <which>_ema），McNemar 配对用哪份各自独立。
    # P1-2（第三轮）：只有 ema_scheme=='step' 才输出 _ema——旧 epoch
    # 粒度方案的污染 shadow 不静默产出无效 EMA 数字（与 resume 判据一致）。
    weights_list = [(ckpt["net"], "")]
    if ckpt.get("net_ema") and ckpt.get("ema_scheme") == "step":
        weights_list.append((ckpt["net_ema"], "_ema"))

    results = []
    for weights, suffix in weights_list:
        # Codex P1-1 修复（2026-10-17）：每份 weights 前重播种——
        # LRHyperNet 的 FPS 用全局 torch.randint，raw 与 EMA 必须共享
        # 同一 FPS 起点（否则配对比较混入 ~0.5-0.9pp 抽样噪声）。
        torch.manual_seed(eval_seed)
        torch.cuda.manual_seed_all(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        model = build_lrh(saved_args, num_category)
        model.load_state_dict(weights)  # strict=True，形状不匹配直接报错
        model.to(device)
        model.eval()

        loader, n_samples = load_test_loader(data_path, num_point, batch_size)

        y_pred, y_true = [], []
        with torch.no_grad():
            for data, label in loader:
                data = data.to(device)
                label = label.to(device).squeeze()
                data = data.permute(0, 2, 1)  # [B, N, 4] -> [B, 4, N]
                logits = model(data)
                preds = logits.max(dim=1)[1]
                y_true.append(label.cpu().numpy())
                y_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = 100.0 * float((y_true == y_pred).sum()) / len(y_true)

        # 输出路径直接对齐 mcnemar_test.py 的读取约定（<preds_dir>/<log_dir>/<which>_seed<S>.npz，
        # 无中间 preds/ 层）——历史 FECNet 版 extract_preds.py 输出 preds/ 层与 mcnemar
        # 读取路径自相矛盾，靠手动桥接才跑通；G16 是新链路，这里直接对齐，出数后
        # `python mcnemar_test.py --preds-dir log/lr_hypernet` 即可开箱即用。
        # EMA 版后缀 _ema：文件名维度完整（which × ema × seed），
        # 与 npz 竞态教训一致（eval_seed 进文件名的同款原因）。
        out_dir = os.path.join("log", "lr_hypernet", log_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "%s%s_seed%d.npz" % (which, suffix, eval_seed))
        np.savez(
            out_path,
            y_true=y_true,
            y_pred=y_pred,
            acc=np.array([acc]),
            log_dir=np.array([log_dir]),
            which=np.array([which + suffix]),
            eval_seed=np.array([eval_seed]),
            epoch=np.array([ckpt.get("epoch", -1)]),
        )
        print("[OK] %-40s %-9s seed=%-4d ckpt_ep=%-3s acc=%.3f n=%d -> %s"
              % (log_dir, which + suffix, eval_seed, ckpt.get("epoch", -1), acc,
                 len(y_true), out_path))
        print("     g16_treatment=%r selector_mode=%r hyperbolic_type=%r"
              % (saved_args.get("g16_treatment"), saved_args.get("selector_mode"),
                 saved_args.get("hyperbolic_type")))
        results.append(acc)
    # raw 版结果为准返回（兼容旧的 results 收集）；EMA 版结果已独立落盘
    return results[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--log_dirs", default=None, help="comma-separated list")
    ap.add_argument("--which", default="last", choices=["last", "best", "both"])
    ap.add_argument("--data_path", default="./ncaltech101_official_gen/")
    ap.add_argument("--num_point", type=int, default=10240)
    ap.add_argument("--num_category", type=int, default=101)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--eval_seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed_all(args.eval_seed)
    np.random.seed(args.eval_seed)
    random.seed(args.eval_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.log_dirs:
        runs = [s.strip() for s in args.log_dirs.split(",") if s.strip()]
    elif args.log_dir:
        runs = [args.log_dir]
    else:
        ap.error("need --log_dir or --log_dirs")

    if args.gpu and args.gpu.lower() != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda", int(args.gpu))
    else:
        device = torch.device("cpu")
    print("device=%s  runs=%d  which=%s  eval_seed=%d"
          % (device, len(runs), args.which, args.eval_seed))

    which_list = ["last", "best"] if args.which == "both" else [args.which]
    results = []
    for ld in runs:
        for w in which_list:
            try:
                acc = extract_one(ld, w, args.data_path, device,
                                  args.num_point, args.num_category,
                                  args.batch_size, args.eval_seed)
                if acc is not None:
                    results.append((ld, w, acc))
            except Exception as exc:
                print("[FAIL] %s/%s: %r" % (ld, w, exc))

    print("\n===== SUMMARY (%d extracted) =====" % len(results))
    for ld, w, acc in results:
        print("%-40s %-4s %.3f" % (ld, w, acc))


if __name__ == "__main__":
    main()
