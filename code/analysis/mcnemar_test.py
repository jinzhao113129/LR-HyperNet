#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired significance testing for SECNet component ablations.

Why this exists
---------------
Reported ablation numbers are single-seed accuracies on a 934-sample test set
(95% CI about +-3.2%), so small differences cannot be trusted from two aggregate
numbers. Worse, FECNet's FPS picks a random start point via torch.randint
(models/modules.py: furthest_point_sample), so evaluation is non-deterministic
unless the RNG is seeded: measured spread on ONE checkpoint is about 2.6pp, i.e.
the same order as every component effect we care about.

Because the test loader uses shuffle=False, the 934 samples are in a FIXED order,
so per-sample predictions from different runs are index-aligned. That permits a
paired test: compare, sample by sample, where component and baseline disagree.

File convention (IMPORTANT)
---------------------------
tools/extract_preds.py writes  <preds-dir>/<log_dir>/last_seed<S>.npz
The eval seed is part of the filename on purpose: several extraction jobs run
concurrently, and a shared "last.npz" name lets one job's seed-7 run silently
overwrite another job's seed-42 run, which mixes eval seeds inside a paired test
and yields a meaningless p-value. That actually happened, so never go back to a
seed-less filename.

Usage
-----
  python mcnemar_test.py --preds-dir <dir> --seeds 42,7,13
  # <dir> holds <log_dir>/last_seed<S>.npz for each seed listed

The FIRST seed is the primary comparison. All seeds are also used for the
cross-seed robustness table and the majority-vote ensemble, which are the
readings to trust.
"""
import argparse
import os
import sys

import numpy as np

WHICH = "last"  # 默认口径（main 里 --which 可覆盖，支持 last/best/last_ema/best_ema）


def load_preds(preds_dir, log_dir, seed, which=None):
    """Load one run's predictions for one eval seed, or None if absent.

    Codex P1 修复（2026-10-17 第三轮）：向后兼容回退——历史 FECNet
    extract 输出在 <log_dir>/preds/ 子层（新代码已统一去掉），先读
    新路径（无 preds/ 层），不存在再回退 <log_dir>/preds/ 旧路径。
    """
    w = which if which is not None else WHICH
    path = os.path.join(preds_dir, log_dir, "%s_seed%s.npz" % (w, seed))
    if not os.path.exists(path):
        path = os.path.join(preds_dir, log_dir, "preds", "%s_seed%s.npz" % (w, seed))
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "y_true": d["y_true"],
        "y_pred": d["y_pred"],
        "acc": float(d["acc"][0]),
        "path": path,
    }


def mcnemar(y_true, pred_a, pred_b):
    """Paired comparison of two classifiers on identical samples.

    b = A correct & B wrong   (A wins the sample)
    c = A wrong & B correct   (B wins the sample)
    Under H0 (equal error rates) b and c are exchangeable: b ~ Binomial(b+c, 0.5).
    """
    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n = len(y_true)

    denom = b + c
    if denom == 0:
        chi2, p_chi2 = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / denom
        p_chi2 = float(chi2_sf(chi2))
    p_exact = float(binom_two_sided(b, denom)) if denom > 0 else 1.0

    return {
        "n": n, "b": b, "c": c,
        "acc_a": 100.0 * float(ok_a.sum()) / n,
        "acc_b": 100.0 * float(ok_b.sum()) / n,
        "chi2": chi2, "p_chi2": p_chi2, "p_exact": p_exact,
    }


def chi2_sf(x):
    """Survival function of chi-square with df=1: P(X > x) = erfc(sqrt(x/2))."""
    from math import erfc, sqrt
    return erfc(sqrt(x / 2.0))


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value (sum of probabilities <= observed)."""
    from math import comb
    if n == 0:
        return 1.0
    probs = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(pr for pr in probs if pr <= obs * (1 + 1e-9)))


def bootstrap_paired(y_true, pred_a, pred_b, n_boot=10000, seed=0):
    """Bootstrap CI for the paired accuracy difference (A - B), resampling samples."""
    rng = np.random.default_rng(seed)
    diff = (pred_a == y_true).astype(np.float64) - (pred_b == y_true).astype(np.float64)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean() * 100.0), float(lo), float(hi)


def majority_vote(pred_list):
    """Per-sample majority vote across several prediction arrays."""
    stack = np.stack(pred_list, axis=0)
    out = np.empty(stack.shape[1], dtype=stack.dtype)
    for i in range(stack.shape[1]):
        vals, counts = np.unique(stack[:, i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--baseline", default="secnet_baseline_match")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated eval seeds; the FIRST is the primary comparison")
    ap.add_argument("--runs", default=None,
                    help="comma-separated log_dirs; default = every dir under preds-dir")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--which", default="last",
                    choices=["last", "best", "last_ema", "best_ema"],
                    help="Codex P1-2 修复（2026-10-17）：读哪个 npz 口径——"
                         "last/best 为 raw，last_ema/best_ema 为 EMA 出口"
                         "（extract_preds 系双输出产物）。默认 last 不变。")
    args = ap.parse_args()

    global WHICH  # 模块级覆盖（load_preds 默认参数读全局）
    WHICH = args.which

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    primary = seeds[0]

    base = load_preds(args.preds_dir, args.baseline, primary)
    if base is None:
        sys.exit("baseline npz not found: %s/%s/%s_seed%s.npz"
                 % (args.preds_dir, args.baseline, WHICH, primary))

    if args.runs:
        runs = [s.strip() for s in args.runs.split(",") if s.strip()]
    else:
        runs = sorted(d for d in os.listdir(args.preds_dir)
                      if os.path.exists(os.path.join(
                          args.preds_dir, d, "%s_seed%s.npz" % (WHICH, primary))))

    print("baseline: %s  seed=%s  acc=%.3f  n=%d"
          % (args.baseline, primary, base["acc"], len(base["y_true"])))
    print("all comparisons below use eval_seed=%s only (mixing seeds invalidates the test)\n"
          % primary)

    hdr = ("%-32s %8s %7s %6s %6s %9s %10s %8s  %s"
           % ("component", "acc", "delta", "win", "lose", "chi2", "p_exact", "p_chi2", "verdict"))
    print(hdr)
    print("-" * len(hdr))
    print("win  = component correct AND baseline wrong  (component wins this sample)")
    print("lose = component wrong   AND baseline correct (baseline wins this sample)")
    print()

    rows = []
    for ld in runs:
        if ld == args.baseline:
            continue
        cur = load_preds(args.preds_dir, ld, primary)
        if cur is None:
            print("%-32s %8s" % (ld, "(no npz for seed %s)" % primary))
            continue
        if not np.array_equal(cur["y_true"], base["y_true"]):
            print("%-32s  !! y_true MISMATCH - index alignment broken, skipping" % ld)
            continue

        r = mcnemar(base["y_true"], cur["y_pred"], base["y_pred"])
        d, lo, hi = bootstrap_paired(base["y_true"], cur["y_pred"], base["y_pred"],
                                     n_boot=args.n_boot)
        print("%-32s %8.3f %+7.3f %6d %6d %9.3f %10.4f %8.4f  %s"
              % (ld, r["acc_a"], d, r["b"], r["c"], r["chi2"],
                 r["p_exact"], r["p_chi2"],
                 "SIGNIFICANT" if r["p_exact"] < 0.05 else "not sig"))
        rows.append((ld, (d, lo, hi)))

    if rows:
        print("\n===== paired bootstrap 95% CI on delta (component - baseline) =====")
        for ld, (d, lo, hi) in rows:
            print("%-32s delta=%+6.3f  95%%CI [%+6.3f, %+6.3f]%s"
                  % (ld, d, lo, hi, "   *CI excludes 0*" if lo > 0 or hi < 0 else ""))

    print("\nNote: 'win' and 'lose' are the only samples carrying paired signal; n00/n11")
    print("      contribute nothing. p_exact is an exact two-sided binomial test")
    print("      (preferred at these counts); p_chi2 adds McNemar's continuity correction.")

    if len(seeds) < 2:
        print("\n(only one eval seed given - cross-seed robustness and the ensemble were skipped)")
        return

    print("\n\n===== CROSS-SEED ROBUSTNESS =====")
    print("FPS resampling makes one eval draw worth ~2.6pp, so a component is only credible")
    print("if its advantage holds across eval seeds.\n")
    head = "%-32s" % "component"
    for s in seeds:
        head += " %14s" % ("seed" + s)
    print(head + "   consistent?")
    print("-" * (len(head) + 16))
    for ld in runs:
        if ld == args.baseline:
            continue
        cells, flags = [], []
        for s in seeds:
            b = load_preds(args.preds_dir, args.baseline, s)
            c = load_preds(args.preds_dir, ld, s)
            if b is None or c is None:
                cells.append("%14s" % "n/a")
                continue
            if not np.array_equal(b["y_true"], c["y_true"]):
                cells.append("%14s" % "MISMATCH")
                continue
            rr = mcnemar(b["y_true"], c["y_pred"], b["y_pred"])
            cells.append("%+7.3f/%.3f" % (rr["acc_a"] - rr["acc_b"], rr["p_exact"]))
            flags.append(rr["p_exact"] < 0.05 and rr["acc_a"] > rr["acc_b"])
        verdict = "YES" if flags and all(flags) else ("mixed" if flags else "no")
        print("%-32s %s   %s" % (ld, " ".join(cells), verdict))
    print("\ncell = <delta acc>/<exact p>. 'consistent?' = improved AND significant on EVERY seed.")

    print("\n\n===== MULTI-SEED MAJORITY-VOTE ENSEMBLE =====")
    print("Aggregating several eval seeds by per-sample majority vote shrinks the FPS noise,")
    print("so this is the most reliable read of the same trained checkpoints.\n")

    def ensemble(log_dir):
        preds, yt = [], None
        for s in seeds:
            p = load_preds(args.preds_dir, log_dir, s)
            if p is None:
                return None, None, 0
            if yt is None:
                yt = p["y_true"]
            elif not np.array_equal(yt, p["y_true"]):
                return None, None, 0
            preds.append(p["y_pred"])
        if len(preds) < 2:
            return None, None, len(preds)
        return majority_vote(preds), yt, len(preds)

    be, byt, bn = ensemble(args.baseline)
    if be is None:
        print("baseline ensemble unavailable (need >=2 eval seeds on disk)")
        return
    print("baseline ensemble over %d eval seeds  acc=%.3f" % (bn, 100.0 * (be == byt).mean()))
    print()
    eh = "%-32s %8s %8s %7s %6s %6s %10s  %s" % (
        "component", "ens_acc", "primary", "delta", "win", "lose", "p_exact", "verdict")
    print(eh)
    print("-" * len(eh))
    for ld in runs:
        if ld == args.baseline:
            continue
        ce, cyt, cn = ensemble(ld)
        single = load_preds(args.preds_dir, ld, primary)
        if ce is None:
            print("%-32s %8s" % (ld, "(insufficient seeds)"))
            continue
        if not np.array_equal(cyt, byt):
            print("%-32s %8s" % (ld, "(y_true mismatch)"))
            continue
        rr = mcnemar(byt, ce, be)
        d = 100.0 * (ce == byt).mean() - 100.0 * (be == byt).mean()
        print("%-32s %8.3f %8.3f %+7.3f %6d %6d %10.4f  %s"
              % (ld, 100.0 * (ce == byt).mean(),
                 single["acc"] if single else float("nan"),
                 d, rr["b"], rr["c"], rr["p_exact"],
                 "SIGNIFICANT" if rr["p_exact"] < 0.05 else "not sig"))
    print("\n'primary' column = the single-seed accuracy; note how far it can sit from the")
    print("ensemble value - that gap is exactly the FPS noise we are trying to control.")


if __name__ == "__main__":
    main()
