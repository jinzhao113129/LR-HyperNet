"""对单个训练样本/中心扫描全部候选，并求小候选池内的精确最优 16 点。

全量 10240 选 16 不可穷举；本工具只声明给定候选池内的最优解。
只读取 train.h5，不训练，也不修改模型或 checkpoint。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from point_attribution_analysis import SubsetIntervention, loss_and_logit
from selector_utility_probe import SelectorIntervention, load_model, predict, train_examples


def checkpoint_hash(args: argparse.Namespace) -> str:
    """返回当前 checkpoint 的 SHA256，防止错用旧扫描缓存。"""
    from selector_utility_probe import ROOT
    path = (ROOT / "log" / "classification" / args.log_dir /
            "checkpoints" / f"{args.ckpt}_checkpoint.pth")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(model, intervention, x, label, seed, indices):
    """在固定 FPS 随机种子下测某组点的完整分类损失。"""
    intervention.selected = list(indices)
    loss, logit = loss_and_logit(model, x, label, seed, x.device)
    if not math.isfinite(loss):
        raise RuntimeError("前向出现非有限 loss")
    return loss, logit


def scan_all(model, intervention, x, label, seed, size, cache, metadata, flush):
    """逐点扫描全部候选，定期保存进度；重新运行可继续。"""
    losses = np.full(size, np.nan, dtype=np.float32)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            old = json.loads(str(saved["metadata"]))
            if old != metadata or len(saved["losses"]) != size:
                raise ValueError("扫描缓存与当前样本/中心/checkpoint 不匹配")
            losses = saved["losses"].copy()
    started = time.perf_counter()
    for index in range(size):
        if np.isfinite(losses[index]):
            continue
        losses[index] = evaluate(model, intervention, x, label, seed, [index])[0]
        if (index + 1) % flush == 0 or index + 1 == size:
            np.savez_compressed(cache, losses=losses,
                                metadata=np.array(json.dumps(metadata)))
            print(f"single scan {index + 1}/{size}", flush=True)
    return losses, time.perf_counter() - started


def make_pool(baseline: np.ndarray, single_losses: np.ndarray,
              pool_size: int) -> list[int]:
    """保留基线 16 点，再补入单点损失最低的候选，保持池定义明确。"""
    base = sorted(set(int(i) for i in baseline))
    if len(base) != 16 or pool_size < 16 or pool_size > len(single_losses):
        raise ValueError("基线必须有 16 个不同点，pool_size 须在 [16,N] 内")
    pool = list(base)
    for index in np.argsort(single_losses, kind="stable"):
        if int(index) not in base:
            pool.append(int(index))
        if len(pool) == pool_size:
            break
    return sorted(pool)


def exact_search(model, intervention, x, label, seed, pool,
                 state_file, metadata, flush):
    """枚举候选池的全部 16 点组合，支持按组合序号续跑。"""
    start, best_loss, best = 0, math.inf, None
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["metadata"] != metadata or state["pool"] != pool:
            raise ValueError("穷举进度与当前候选池不匹配")
        start, best_loss, best = state["next"], state["best_loss"], state["best"]
    total = math.comb(len(pool), 16)
    started = time.perf_counter()
    for number, subset in enumerate(itertools.combinations(pool, 16)):
        if number < start:
            continue
        loss, _ = evaluate(model, intervention, x, label, seed, subset)
        if loss < best_loss:
            best_loss, best = loss, list(subset)
        if (number + 1) % flush == 0 or number + 1 == total:
            progress = {"metadata": metadata, "pool": pool, "next": number + 1,
                        "best_loss": best_loss, "best": best, "total": total}
            state_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(f"exact pool {number + 1}/{total}", flush=True)
    return best, best_loss, time.perf_counter() - started


def removal_check(model, intervention, x, label, seed, chosen, full_loss):
    """逐一移除最终集合中的点，测该点的条件贡献。"""
    result = []
    for index in chosen:
        reduced = [item for item in chosen if item != index]
        loss, _ = evaluate(model, intervention, x, label, seed, reduced)
        result.append({"index": index, "loss_increase": loss - full_loss})
    return result


def save_pool_features(path, intervention, selector, center, pool,
                       single_losses):
    """保存池内点和局部邻居的几何/中间特征，供后续解释分析。"""
    ids = torch.as_tensor(pool, device=selector.all_feat.device)
    local = selector.local_idx[0, center]
    origin = intervention.fps_idx[0, center]
    xyz = intervention.input_xyz[0]
    feature = selector.all_feat[0]
    arr = lambda tensor: tensor.detach().cpu().numpy().astype(np.float16)
    np.savez_compressed(
        path, pool_indices=np.asarray(pool, dtype=np.int32),
        pool_xyz=arr(xyz[ids]), pool_feature=arr(feature[ids]),
        local_indices=local.detach().cpu().numpy(),
        local_xyz=arr(xyz[local]), local_feature=arr(feature[local]),
        center_xyz=arr(xyz[origin]), center_feature=arr(feature[origin]),
        single_losses=single_losses)


def pilot_cost(model, intervention, x, label, seed, size, count):
    """均匀抽取候选估计完整前向吞吐量与 CUDA 峰值。"""
    ids = np.linspace(0, size - 1, min(count, size), dtype=np.int64)
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats(x.device)
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    for index in ids:
        evaluate(model, intervention, x, label, seed, [int(index)])
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(x.device) if x.is_cuda else None
    return {"pilot_points": len(ids), "seconds": seconds,
            "estimated_single_scan_hours": seconds / len(ids) * size / 3600,
            "cuda_peak_allocated_bytes": peak}


def run(args: argparse.Namespace) -> None:
    """加载固定模型并串行执行全点扫描、池内穷举和移除复核。"""
    if args.pool_size < 16 or args.pool_size > args.num_point:
        raise ValueError("pool_size 必须在 [16,num_point] 内")
    total = math.comb(args.pool_size, 16)
    if not args.pilot_points and total > args.max_combinations:
        raise ValueError(f"池内共有 {total:,} 个组合，超过预算上限")
    if args.flush < 1:
        raise ValueError("flush 必须为正数")
    if args.pilot_points < 0:
        raise ValueError("pilot_points 必须 >= 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    data, labels = train_examples(args)
    x = torch.from_numpy(data[0]).float().T.unsqueeze(0).to(device)
    label = int(labels[0])
    grouper = model.local_grouper_list[args.stage]
    if grouper.remote_selector.g16_treatment in {"random", "spatial_fixed"}:
        raise ValueError("需要确定性的固定 G16 checkpoint")
    if grouper.use_remote_gate or grouper.use_global_anchor or grouper.remote_kda_enc:
        raise ValueError("当前干预不支持 gate/anchor/remote KDA")
    selector = SelectorIntervention(grouper.remote_selector)
    selector.install()
    intervention = SubsetIntervention(grouper, selector)
    pre = grouper.register_forward_pre_hook(intervention.capture_input)
    post = grouper.register_forward_hook(intervention.hook)
    try:
        result = search_loaded(args, model, selector, intervention, x, label)
    finally:
        pre.remove()
        post.remove()
        selector.remove()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def build_result(model, intervention, x, label, seed, metadata, pool,
                 baseline, baseline_loss, empty_loss, best, best_loss,
                 scan_time, exact_time):
    """复核池内最优结果并整理可解释分析所需的数值。"""
    verified, _ = evaluate(model, intervention, x, label, seed, best)
    if abs(verified - best_loss) > 1e-4:
        raise RuntimeError("最优集合重复前向结果不一致")
    return {"metadata": metadata, "candidate_scope": "all N for single scan",
            "optimal_scope": f"exact within this {len(pool)}-point pool only",
            "pool": pool, "baseline": baseline.tolist(), "best": best,
            "baseline_loss": baseline_loss, "empty_loss": empty_loss,
            "best_loss": best_loss, "gain_vs_baseline": baseline_loss - best_loss,
            "removal": removal_check(model, intervention, x, label, seed,
                                      best, best_loss),
            "seconds": {"single_scan": scan_time, "exact_pool": exact_time}}


def search_loaded(args, model, selector, intervention, x, label):
    """从已加载模型取得基线 16 点，再运行扫描与精确池搜索。"""
    seed = args.seed + args.sample_id
    intervention.center = None
    predict(model, x, seed, x.device)
    if args.center < 0 or args.center >= selector.center_feat.shape[1]:
        raise ValueError("center 超出该 stage 的 FPS 中心范围")
    intervention.center = args.center
    baseline = selector.baseline_idx[0, args.center].cpu().numpy()
    metadata = {"checkpoint_sha256": checkpoint_hash(args),
                "log_dir": args.log_dir, "ckpt": args.ckpt,
                "weight": "ema" if args.use_ema else "raw",
                "split": "train", "sample_id": args.sample_id,
                "stage": args.stage, "center": args.center, "seed": seed}
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"s{args.sample_id}_stage{args.stage}_c{args.center}"
    if args.pilot_points:
        result = {"metadata": metadata, "pilot": pilot_cost(
            model, intervention, x, label, seed,
            selector.all_feat.shape[1], args.pilot_points)}
        (out / f"{stem}_pilot.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    scan_cache = out / f"{stem}_single_scan.npz"
    state_file = out / f"{stem}_pool{args.pool_size}_progress.json"
    baseline_loss, _ = evaluate(model, intervention, x, label, seed, baseline)
    empty_loss, _ = evaluate(model, intervention, x, label, seed, [])
    losses, scan_time = scan_all(model, intervention, x, label, seed,
                                 selector.all_feat.shape[1], scan_cache,
                                 metadata, args.flush)
    pool = make_pool(baseline, losses, args.pool_size)
    save_pool_features(out / f"{stem}_pool_features.npz", intervention,
                       selector, args.center, pool, losses)
    best, best_loss, exact_time = exact_search(
        model, intervention, x, label, seed, pool, state_file,
        metadata, args.flush)
    result = build_result(model, intervention, x, label, seed, metadata,
                          pool, baseline, baseline_loss, empty_loss, best,
                          best_loss, scan_time, exact_time)
    (out / f"{stem}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    """读取单样本单中心诊断参数；默认池 20 点，即 4845 个组合。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", default="secnet_pi2_g16_cosine_baidu")
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--num_point", type=int, default=10240)
    parser.add_argument("--num_category", type=int, default=101)
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--center", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--max_combinations", type=int, default=5000)
    parser.add_argument("--pilot_points", type=int, default=0,
                        help="若大于 0，仅抽点测量全点扫描成本，不执行搜索")
    parser.add_argument("--flush", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="~/zhaojin/logs_run/point_subset_search")
    args = parser.parse_args()
    args.sample_ids = [args.sample_id]
    args.max_samples = 1
    return args


if __name__ == "__main__":
    run(parse_args())
