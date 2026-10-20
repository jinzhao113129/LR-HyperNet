#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-sample test predictions for paired significance testing.

Loads each experiment's checkpoint, rebuilds the model from the checkpoint's
own saved `args` (so component flags are guaranteed identical to the original
run), runs inference on the standard test split, and saves an .npz per run.

Output per run:  log/classification/<log_dir>/preds/<which>.npz
  keys: y_true (934,), y_pred (934,), acc (scalar), log_dir, which, epoch

Usage:
  python3 extract_preds.py --log_dir secnet_baseline_match --which last
  python3 extract_preds.py --log_dirs all --which both
"""
import argparse
import os
import random
import re
import sys

import numpy as np
import torch

# This script lives in <repo>/tools/, so the repo root is the parent of its dir.
# provider_data.py lives in the repo root; models/FECNet.py does `from modules import ...`
# so models/ must be importable too (the sbatch scripts achieve this via PYTHONPATH).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))

import provider_data  # noqa: E402
from models.FECNet import FECNet  # noqa: E402

# All 60-epoch standard-ablation experiments eligible for the paired test.
STD_RUNS = [
    "secnet_baseline_match",
    "secnet_timeenc_v100",
    "secnet_rp1_v100",
    "secnet_hyperclf_v100",
    "secnet_hyperfusion_v100",
    "secnet_hyperattn_v100",
    "secnet_hyperembed_v100",
    "secnet_hyper_selector_v100",
    "secnet_timeenc_rp1_v100",
    "secnet_mamba_v100",
    "secnet_hyperembed_fix_v100",
]

FLAG_KEYS = [
    "group_norm", "time_enc", "kda", "rp1_selector", "hyperbolic_selector",
    "spike_time_enc", "hyperbolic_classifier", "hyperbolic_fusion",
    "hyperbolic_embedding", "hyperbolic_attention", "event_mamba_ssm",
    # 2026-10-14 ai2 补齐：scale_norm_fps 改变 LocalGrouper forward 路径（归一化
    # 替代 fps_parameter 缩放），hyperbolic_mlp 增删双曲 MLP 模块。两者缺失时重建
    # 的模型与训练时不一致，推理结果无效（scale_norm 首次推理 2.57% 即此 bug）。
    "hyperbolic_mlp", "scale_norm_fps",
    # 2026-10-14 ai2 追加：hsel_scaled_coords 改变 HyperbolicUniformGrouper 的
    # 选点坐标口径（缩放后 vs 原始），缺失时按 False 重建会得到错误的
    # "双曲+缩放坐标"实验结果。
    "hsel_scaled_coords",
    # 2026-10-18 ai2：hyperbolic_gate / fca_gate 增删 hyper_gates/fca_gates 模块，
    # 且替换 gated fusion 的 forward 路径。缺失时按 False 重建会用从未训练的
    # group_conv/bn_global 推理，结果静默错误（同 scale_norm 那次的 bug 模式）。
    "hyperbolic_gate", "fca_gate",
    # 2026-10-18：hyperbolic_radial 增删 hyper_radial 模块 + 改 embed 后 forward 路径
    "hyperbolic_radial", "polar_feature",
]


def parse_flags_from_log(log_dir):
    """Recover the component flags from the run's own training log.

    The 11 runs that predate the `args=vars(args)` checkpoint change store no args
    in their .pth. The very first line of logs/pointmlp.txt is the full
    `Namespace(...)` dump, which is the authoritative record of what that run used.

    Keys absent from an older log are treated as False, which matches the argparse
    `default=False` of every use_* flag in train_frequency_ncaltech.py.
    """
    log_path = os.path.join("log", "classification", log_dir, "logs", "pointmlp.txt")
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, errors="replace") as fh:
        for line in fh:
            if "Namespace(" in line:
                body = line[line.index("Namespace(") + len("Namespace("):]
                body = body.rsplit(")", 1)[0]
                found = {}
                for key in FLAG_KEYS:
                    found[key] = bool(re.search(r"\b%s=True\b" % re.escape(key), body))
                # also recover numeric/data settings for a sanity cross-check
                meta = {}
                for key in ("num_point", "num_category", "batch_size", "data_path"):
                    m = re.search(r"\b%s=([^,]+)" % re.escape(key), body)
                    if m:
                        meta[key] = m.group(1).strip().strip("'\"")
                return found, meta
    return None, None


def build_model(saved_args, num_category):
    """Rebuild FECNet using exactly the component flags recorded for the run."""
    # 2026-10-14 ai2 补齐：新增 flag 在旧 checkpoint/旧日志里可能不存在
    # （argparse default=False），与 parse_flags_from_log 的兜底策略一致：缺键视为 False。
    defaults = {"hyperbolic_mlp": False, "scale_norm_fps": False,
                "hsel_scaled_coords": False,
                "hyperbolic_gate": False, "fca_gate": False,
                "hyperbolic_radial": False, "polar_feature": False}
    for key, val in defaults.items():
        if key not in saved_args:
            saved_args[key] = val
            print("[INFO] flag '%s' missing in saved args -> treating as False" % key)
    missing = [k for k in FLAG_KEYS if k not in saved_args]
    if missing:
        raise ValueError("flags missing keys: %s" % missing)
    return FECNet(
        num_classes=num_category,
        use_group_norm=saved_args["group_norm"],
        use_time_enc=saved_args["time_enc"],
        use_kda=saved_args["kda"],
        use_rp1_selector=saved_args["rp1_selector"],
        use_hyperbolic_selector=saved_args["hyperbolic_selector"],
        use_spike_time_enc=saved_args["spike_time_enc"],
        use_hyperbolic_classifier=saved_args["hyperbolic_classifier"],
        use_hyperbolic_fusion=saved_args["hyperbolic_fusion"],
        use_hyperbolic_embedding=saved_args["hyperbolic_embedding"],
        use_hyperbolic_attention=saved_args["hyperbolic_attention"],
        use_event_mamba_ssm=saved_args["event_mamba_ssm"],
        use_hyperbolic_mlp=saved_args["hyperbolic_mlp"],
        use_scale_norm_fps=saved_args["scale_norm_fps"],
        use_hsel_scaled_coords=saved_args["hsel_scaled_coords"],
        use_hyperbolic_gate=saved_args["hyperbolic_gate"],
        use_fca_gate=saved_args["fca_gate"],
        use_hyperbolic_radial=saved_args["hyperbolic_radial"],
        use_polar_feature=saved_args["polar_feature"],
    ), {k: saved_args[k] for k in FLAG_KEYS}


def load_test_loader(data_path, num_point, batch_size):
    """Identical to the training script: TensorDataset + shuffle=False.

    shuffle=False is what makes per-sample predictions comparable across runs.
    """
    test_file = data_path + "test.h5"
    data, label, _mark = provider_data.load_h5_mark(test_file)
    label = np.squeeze(label)
    data = torch.from_numpy(data).reshape(-1, num_point, 4)
    label = torch.from_numpy(label.astype("int64"))
    ds = torch.utils.data.TensorDataset(data, label)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    ), data.shape[0]


def extract_one(log_dir, which, data_path, device, num_point, num_category,
                batch_size, eval_seed):
    # 2026-10-14 ai2 修复：每个 (log_dir, which) 独立重播种。
    # 之前只在 main() 播种一次：--which both 时 best 的推理发生在 last 的
    # FPS torch.randint 与模型构建的 RNG 消耗之后，同一 checkpoint 的 best
    # 结果随调用方式（单独 best / both / 多目录同进程）漂移，逐样本不同
    # （实测 48.929 vs 49.251）。FPS 起始点 randint 依赖 RNG 状态，必须每轮
    # 重置才能让 (log_dir, which, eval_seed) 唯一决定结果。
    # last 值不受影响（both 模式 last 排第一，状态本就是纯净 seed）。
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)
    ckpt_name = "last_checkpoint.pth" if which == "last" else "best_checkpoint.pth"
    ckpt_path = os.path.join("log", "classification", log_dir, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print("[SKIP] %s: no %s" % (log_dir, ckpt_name))
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Preferred source: args stored in the checkpoint itself (post-2026-10-13 runs).
    saved_args = ckpt.get("args")
    source = "ckpt.args"
    meta = None
    if saved_args is None:
        # Fallback: the 11 pre-existing runs have no args in the .pth, so recover the
        # flags from the run's own log. This is the authoritative record for those runs.
        saved_args, meta = parse_flags_from_log(log_dir)
        source = "log Namespace"
        if saved_args is None:
            print("[SKIP] %s: no args in checkpoint and no Namespace line in log" % log_dir)
            return None

    # Cross-check the reshape/loader settings against what the run actually used.
    if meta:
        for key, want in (("num_point", str(num_point)),
                          ("num_category", str(num_category)),
                          ("batch_size", str(batch_size))):
            got = meta.get(key)
            if got is not None and got != want:
                raise ValueError(
                    "%s: %s mismatch - run used %s, script has %s"
                    % (log_dir, key, got, want))

    model, flags = build_model(saved_args, num_category)

    # Codex P0-2 修复（2026-10-17）：raw/EMA 双输出——net_ema 键存在时
    # 额外评一份（EMA 是并行出口，raw 输出完全不变）。每份 weights 前重
    # 播种（P1-1）：FECNet 的 FPS 用全局 torch.randint，raw 与 EMA 必须
    # 共享同一 FPS 起点否则配对比较混入 ~0.5-0.9pp 抽样噪声。
    # P1-2（第三轮）：只有 ema_scheme=='step' 的 net_ema 才输出 _ema——
    # 旧 epoch 粒度方案的 shadow 已污染（BN stats 陈旧），静默产出无效
    # EMA 数字，与 resume 判据保持一致。
    weights_list = [(ckpt["net"], "")]
    if ckpt.get("net_ema") and ckpt.get("ema_scheme") == "step":
        weights_list.append((ckpt["net_ema"], "_ema"))

    results = []
    for weights, suffix in weights_list:
        # P1-1：每份 weights 重新播种——raw 与 EMA 用同一 FPS 起点
        torch.manual_seed(eval_seed)
        torch.cuda.manual_seed_all(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        try:
            model.load_state_dict(weights)
        except RuntimeError as exc:
            # 兼容旧结构 checkpoint（已确认的实际案例：旧版 --event_mamba_ssm 每 stage
            # 各建一个 MambaSSM（132/268/540 维，forward 在 stage3 用 mamba_ssms[2]），
            # 新代码只建一个 mamba_ssms.0（540 维）。故：
            #   mamba_ssms.2.*  →  改名 mamba_ssms.0.*（真正被 forward 用过的那份）
            #   mamba_ssms.0/1.*（旧版死权重）→ 丢弃
            # 其余多出来的 ckpt 键一律丢弃；缺键绝不补。改名后逐键校验形状，
            # 对不上就报错退出，不做任何静默填充。
            model_sd = model.state_dict()
            model_keys = set(model_sd.keys())
            ckpt_keys = set(weights.keys())
            missing = sorted(model_keys - ckpt_keys)
            if missing:
                raise ValueError("%s: checkpoint 缺少模型必需键 %s（不能容忍）"
                                 % (log_dir, missing[:5]))
            remap = {k: k.replace("mamba_ssms.2.", "mamba_ssms.0.", 1)
                     for k in ckpt_keys if k.startswith("mamba_ssms.2.")}
            for old, new in remap.items():
                if new in model_keys:
                    ck_v, md_v = weights[old], model_sd[new]
                    if tuple(ck_v.shape) != tuple(md_v.shape):
                        raise ValueError("%s: %s 形状 %s 与模型 %s 不符，拒绝加载"
                                         % (log_dir, old, tuple(ck_v.shape),
                                            tuple(md_v.shape)))
            filtered = {}
            for k, v in weights.items():
                if k in remap:
                    continue
                if k in model_keys:
                    filtered[k] = v
            for old, new in remap.items():
                if new in model_keys:
                    filtered[new] = weights[old]
            dropped = sorted((ckpt_keys - model_keys) - set(remap.keys()))
            print("[WARN] %s: 兼容旧结构——%d 个键改名 mamba_ssms.2→mamba_ssms.0，"
                  "丢弃 %d 个死权重键（旧版建3个SSM只用1个）: %s%s"
                  % (log_dir, len(remap), len(dropped), ", ".join(dropped[:6]),
                     "..." if len(dropped) > 6 else ""))
            model.load_state_dict(filtered)
        model.to(device)
        model.eval()
        model.apply(lambda m: setattr(m, "inplace", True)
                    if m.__class__.__name__.find("ReLU") != -1 else None)

        loader, n_samples = load_test_loader(data_path, num_point, batch_size)

        y_pred, y_true = [], []
        with torch.no_grad():
            for data, label in loader:
                data = data.to(device)
                label = label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits = model(data)
                preds = logits.max(dim=1)[1]
                y_true.append(label.cpu().numpy())
                y_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = 100.0 * float((y_true == y_pred).sum()) / len(y_true)

        out_dir = os.path.join("log", "classification", log_dir)
        os.makedirs(out_dir, exist_ok=True)
        # Codex P1 修复（2026-10-17 第三轮）：去掉中间 preds/ 层，对齐
        # mcnemar 读取约定 <preds_dir>/<log_dir>/<which>_seed<S>.npz——
        # 历史上 FECNet 版靠手动搬运才跑通，且 --which last_ema 因此只对
        # LRHyperNet 生效；现统一无 preds/ 层（历史 npz 已在 preds/ 下，
        # mcnemar 侧做向后兼容回退读取）。
        # CRITICAL: the filename MUST encode the eval seed. Several extraction jobs run
        # concurrently, and a shared "<which>.npz" name means a job using seed 7 silently
        # overwrites the seed-42 file of another job - which silently mixes eval seeds in a
        # later paired test and produces a meaningless p-value. (This actually happened.)
        # EMA 版带 _ema 后缀（which × ema × seed 文件名维度完整，同款防覆盖逻辑）。
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
        print("[OK] %-34s %-9s seed=%-4d ckpt_ep=%-3s acc=%.3f n=%d (flags from %s) -> %s"
              % (log_dir, which + suffix, eval_seed, ckpt.get("epoch", -1), acc,
                 len(y_true), source, out_path))
        print("     flags: %s" % flags)
        results.append(acc)
    # raw 版结果为准返回（兼容旧的 results 收集）；EMA 版已独立落盘
    return results[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--log_dirs", default=None,
                    help="'all' for the standard 60-epoch set, else comma-separated")
    ap.add_argument("--which", default="last", choices=["last", "best", "both"])
    ap.add_argument("--data_path", default="./ncaltech101_official_baidu/")
    ap.add_argument("--num_point", type=int, default=10240)
    ap.add_argument("--num_category", type=int, default=101)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--eval_seed", type=int, default=42,
                    help="fixed seed set before inference; FPS uses torch.randint, "
                         "so a fixed seed makes per-sample predictions deterministic "
                         "and lets every component share the SAME group centers")
    args = ap.parse_args()

    # CRITICAL: FECNet's FPS (models/modules.py furthest_point_sample) starts from
    # torch.randint, so inference is non-deterministic unless the RNG is seeded here.
    # Seeding BEFORE any torch op guarantees all runs (and all components) use the
    # same deterministic group centers -> valid paired comparison.
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed_all(args.eval_seed)
    np.random.seed(args.eval_seed)
    random.seed(args.eval_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.log_dirs == "all":
        runs = STD_RUNS
    elif args.log_dirs:
        runs = [s.strip() for s in args.log_dirs.split(",") if s.strip()]
    elif args.log_dir:
        runs = [args.log_dir]
    else:
        ap.error("need --log_dir, --log_dirs, or --log_dirs all")

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
            except Exception as exc:  # keep going, report at the end
                print("[FAIL] %s/%s: %r" % (ld, w, exc))

    print("\n===== SUMMARY (%d extracted) =====" % len(results))
    for ld, w, acc in results:
        print("%-34s %-4s %.3f" % (ld, w, acc))


if __name__ == "__main__":
    main()