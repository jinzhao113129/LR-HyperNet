"""在训练集上测量 G16 候选点的实际分类贡献。

每次只替换一个中心的一个远程点，模型、聚合器和其余中心保持不变。
先扫描全部候选的廉价特征并测量耗时，再用完整前向评估混合候选池。
标签只从 train.h5 读取；此脚本不训练，也不读取 test.h5。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "tools")]

import extract_preds  # noqa: E402
import provider_data  # noqa: E402
from modules import furthest_point_sample  # noqa: E402


def sync(device: torch.device) -> None:
    """同步 CUDA，使计时包含实际 GPU 计算。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """从指定 checkpoint 重建模型，并强制校验为固定 G16 结构。"""
    path = (ROOT / "log" / "classification" / args.log_dir / "checkpoints"
            / f"{args.ckpt}_checkpoint.pth")
    checkpoint = torch.load(path, map_location="cpu")
    saved_args = dict(checkpoint["args"])
    if not saved_args.get("remote_g16"):
        raise ValueError("诊断需要 remote_g16=True 的 checkpoint")
    if saved_args.get("g16_treatment") in {"learned", "learned_v2", "csnet"}:
        raise ValueError("诊断要求固定 G16 checkpoint，以便保持原双曲聚合器")
    model, _ = extract_preds.build_model(saved_args, args.num_category)
    key = "net_ema" if args.use_ema else "net"
    if key not in checkpoint or checkpoint[key] is None:
        raise ValueError(f"checkpoint 缺少 {key}")
    model.load_state_dict(checkpoint[key], strict=True)
    return model.to(device).eval()


def train_examples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """只从原始训练文件提取指定样本，不接触测试集标签。"""
    data, labels, _ = provider_data.load_h5_mark(str(ROOT / args.data_path / "train.h5"))
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    ids = np.arange(len(labels)) if args.sample_ids is None else np.array(args.sample_ids)
    if np.any(ids < 0) or np.any(ids >= len(labels)):
        raise ValueError("sample_ids 超出 train.h5 范围")
    ids = ids[:args.max_samples]
    return np.asarray(data)[ids].reshape(-1, args.num_point, 4), labels[ids]


class SelectorIntervention:
    """拦截一个 stage 的固定选择器，只在指定中心替换索引。"""

    def __init__(self, selector: torch.nn.Module):
        """保存原 forward，建立可撤销的诊断状态。"""
        self.selector = selector
        self.original = selector.forward
        self.center: int | None = None
        self.replacement: torch.Tensor | None = None
        self.baseline_idx: torch.Tensor | None = None
        self.center_feat: torch.Tensor | None = None
        self.all_feat: torch.Tensor | None = None
        self.local_idx: torch.Tensor | None = None

    def install(self) -> None:
        """安装单次进程内的 forward 包装，不修改模型文件。"""
        def wrapped(center_feat, all_feat, local_idx, *rest, **kwargs):
            selected = self.original(center_feat, all_feat, local_idx, *rest, **kwargs)
            self.center_feat = center_feat.detach()
            self.all_feat = all_feat.detach()
            self.local_idx = local_idx.detach()
            self.baseline_idx = selected.detach()
            if self.center is None:
                return selected
            changed = selected.clone()
            changed[:, self.center, :] = self.replacement
            return changed
        self.selector.forward = wrapped

    def remove(self) -> None:
        """恢复原 selector.forward。"""
        self.selector.forward = self.original


def predict(model: torch.nn.Module, x: torch.Tensor, seed: int,
            device: torch.device) -> torch.Tensor:
    """以相同 FPS 随机种子执行完整模型前向。"""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        return model(x)


def scores_for_all(center: torch.Tensor, all_feat: torch.Tensor,
                   local_idx: torch.Tensor, remote_idx: torch.Tensor,
                   center_xyz: torch.Tensor, all_xyz: torch.Tensor,
                   polarity: torch.Tensor) -> dict[str, torch.Tensor]:
    """一次计算全部候选的距离、时间和空间新颖性。"""
    diff = all_feat - center
    l2 = diff.square().sum(-1).sqrt()
    cosine = F.cosine_similarity(all_feat, center.expand_as(all_feat), dim=-1)
    delta = all_xyz - center_xyz
    local_xy = all_xyz[local_idx, 1:3]
    novelty = torch.cdist(all_xyz[:, 1:3].unsqueeze(0),
                          local_xy.unsqueeze(0)).squeeze(0).min(-1).values
    valid = torch.ones(len(all_feat), dtype=torch.bool, device=all_feat.device)
    valid[local_idx] = False
    remote_set = torch.zeros_like(valid)
    remote_set[remote_idx] = True
    return {
        "l2": l2, "cosine": cosine, "novelty": novelty,
        "dt": delta[:, 0].abs(), "polarity": polarity,
        "valid": valid, "remote_set": remote_set,
    }


def candidate_pool(feat: dict[str, torch.Tensor], pool_size: int,
                   rng: np.random.Generator) -> list[int]:
    """从近邻、空间新颖、时间跨度、随机四路构建去重池。"""
    valid = torch.where(feat["valid"] & ~feat["remote_set"])[0]
    n = min(pool_size, len(valid))
    quota = max(1, math.ceil(n / 4))
    chosen: list[int] = []
    for key, descending in (("l2", False), ("novelty", True), ("dt", True)):
        order = torch.argsort(feat[key][valid], descending=descending)[:quota]
        chosen.extend(valid[order].cpu().tolist())
    chosen.extend(rng.permutation(valid.cpu().numpy())[:quota].tolist())
    seen: set[int] = set()
    result = [i for i in chosen if not (i in seen or seen.add(i))]
    if len(result) < n:
        for i in rng.permutation(valid.cpu().numpy()).tolist():
            if i not in seen:
                result.append(i)
                seen.add(i)
            if len(result) >= n:
                break
    return result[:n]


def fixed_rule_indices(feat: dict[str, torch.Tensor], k: int,
                       rng: np.random.Generator, rp1_lambda: float) -> dict[str, torch.Tensor]:
    """在相同全候选池和局部排除条件下生成四种固定 16 点对照。"""
    valid = torch.where(feat["valid"])[0]
    if len(valid) < k:
        raise ValueError("非局部候选不足 16 个")
    rp1 = -feat["l2"].square() + rp1_lambda * feat["novelty"]
    return {
        "random": valid[torch.as_tensor(rng.choice(len(valid), k, replace=False),
                                         device=valid.device)].sort().values,
        "euc_near": valid[feat["l2"][valid].topk(k, largest=False).indices].sort().values,
        "rp1_scorer": valid[rp1[valid].topk(k).indices].sort().values,
        "cosine_near": valid[feat["cosine"][valid].topk(k).indices].sort().values,
    }


def evaluate(model: torch.nn.Module, hook: SelectorIntervention,
             x: torch.Tensor, label: int, selected: torch.Tensor,
             center: int, seed: int, device: torch.device) -> tuple[float, float]:
    """完整前向评估替换后的 CE 与正确类别 margin。"""
    hook.center = center
    hook.replacement = selected
    logits = predict(model, x, seed, device)
    target = torch.tensor([label], device=device)
    loss = F.cross_entropy(logits, target).item()
    correct = logits[0, label]
    other = logits[0].clone()
    other[label] = -torch.inf
    margin = (correct - other.max()).item()
    return loss, margin


def pca_summary(rows: list[dict], features: list[np.ndarray]) -> dict:
    """对贡献最高和最低的候选特征差向量统计 PCA。"""
    if len(rows) < 4:
        return {"status": "too_few_candidates"}
    order = np.argsort([r["loss_gain"] for r in rows])
    count = max(2, len(rows) // 4)
    output = {}
    for name, ids in (("low", order[:count]), ("high", order[-count:])):
        matrix = np.stack([features[i] for i in ids]).astype(np.float64)
        matrix -= matrix.mean(0, keepdims=True)
        singular = np.linalg.svd(matrix, compute_uv=False) ** 2
        ratio = singular / max(singular.sum(), 1e-12)
        output[name] = ratio[:5].round(5).tolist()
    return output


def greedy_swaps(model: torch.nn.Module, hook: SelectorIntervention,
                 x: torch.Tensor, label: int, original: torch.Tensor,
                 center: int, seed: int, device: torch.device,
                 ranked: list[dict], top: int, rounds: int,
                 baseline_loss: float) -> dict:
    """在高贡献候选与 16 个槽位上做有限次贪心替换并完整复核。"""
    current = original.clone()
    loss_now = baseline_loss
    changes: list[dict] = []
    best_candidates = [r["candidate"] for r in
                       sorted(ranked, key=lambda r: r["loss_gain"], reverse=True)[:top]]
    for _ in range(rounds):
        best = None
        for candidate in best_candidates:
            if candidate in current.tolist():
                continue
            for slot in range(len(current)):
                trial = current.clone()
                previous = int(trial[slot])
                trial[slot] = candidate
                trial_loss, _ = evaluate(model, hook, x, label, trial,
                                         center, seed, device)
                if trial_loss < loss_now and (best is None or trial_loss < best[0]):
                    best = (trial_loss, candidate, slot, previous)
        if best is None:
            break
        loss_now, candidate, slot, previous = best
        current[slot] = candidate
        changes.append({"slot": slot, "out": previous, "in": candidate,
                        "loss_after": loss_now})
    verified_loss, _ = evaluate(model, hook, x, label, current,
                                center, seed, device)
    if abs(verified_loss - loss_now) > 1e-4:
        raise RuntimeError("相同种子下重复前向不一致；贪心结果不可用")
    return {"changes": changes, "selected_16": current.cpu().tolist(),
            "verified_loss": verified_loss,
            "verified_loss_gain": baseline_loss - verified_loss}


def run(args: argparse.Namespace) -> None:
    """运行小样本全点特征扫描、替换搜索并输出可复核结果。"""
    if args.stage not in (0, 1, 2):
        raise ValueError("stage 仅支持 0、1、2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args, device)
    data, labels = train_examples(args)
    grouper = model.local_grouper_list[args.stage]
    hook = SelectorIntervention(grouper.remote_selector)
    hook.install()
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, inputs):
        """重放 FPS 中心索引，并恢复 RNG 以保持原 forward 不变。"""
        xytp = inputs[0].detach()
        captured["xytp"] = module.fps_parameter * xytp
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(xytp.device) if xytp.is_cuda else None
        if getattr(module, "use_scale_norm_fps", False):
            fps_input = ((xytp - xytp.mean(1, keepdim=True)) /
                         (xytp.std(1, keepdim=True) + 1e-5))
        else:
            fps_input = module.fps_parameter * xytp
        indices = furthest_point_sample(fps_input.contiguous()[:, :, :3],
                                        module.groups).long()
        captured["fps_idx"] = indices.sort(dim=1).values
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, xytp.device)

    handle = grouper.register_forward_pre_hook(pre_hook)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    summaries: list[dict] = []
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for sample_pos, (raw, label) in enumerate(zip(data, labels)):
            x = torch.from_numpy(raw).float().permute(1, 0).unsqueeze(0).to(device)
            seed = args.seed + sample_pos
            hook.center = None
            hook.replacement = None
            logits = predict(model, x, seed, device)
            base_loss = F.cross_entropy(logits, torch.tensor([label], device=device)).item()
            other_logits = logits[0].clone()
            other_logits[label] = -torch.inf
            base_margin = (logits[0, label] - other_logits.max()).item()
            original = hook.baseline_idx.clone()
            features = hook.all_feat[0].clone()
            local = hook.local_idx[0].clone()
            centers = hook.center_feat[0].clone()
            xyz = captured["xytp"][0].clone()
            group = min(args.centers_per_sample, len(centers))
            center_ids = rng.choice(len(centers), size=group, replace=False)
            for center_id in center_ids:
                center_id = int(center_id)
                selected = original[0, center_id].clone()
                fps_index = captured["fps_idx"][0, center_id]
                if not torch.allclose(features[fps_index], centers[center_id]):
                    raise RuntimeError("FPS 中心重放与原 forward 不一致")
                center_xyz = xyz[fps_index]
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sync(device)
                started = time.perf_counter()
                feat = scores_for_all(centers[center_id], features,
                                      local[center_id], selected,
                                      center_xyz[:3], xyz[:, :3], xyz[:, 3])
                sync(device)
                feature_seconds = time.perf_counter() - started
                rule_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, sample_pos, center_id, 1]))
                rules = fixed_rule_indices(feat, len(selected), rule_rng,
                                           grouper.remote_selector.rp1_lambda)
                rule_results = {}
                for name, indices in rules.items():
                    rule_loss, rule_margin = evaluate(
                        model, hook, x, int(label), indices, center_id, seed, device)
                    rule_results[name] = {
                        "loss": rule_loss, "loss_gain": base_loss - rule_loss,
                        "margin_gain": rule_margin - base_margin,
                        "selected_16": indices.cpu().tolist(),
                        "overlap_with_baseline": len(
                            set(indices.cpu().tolist()) & set(selected.cpu().tolist())),
                    }
                pool = (torch.where(feat["valid"] & ~feat["remote_set"])[0].cpu().tolist()
                        if args.scan_all else candidate_pool(feat, args.pool_size, rng))
                # 实测若干完整前向，再按实测速度估算全部候选替换时间。
                probe_count = min(args.probe_count, len(pool))
                start = time.perf_counter()
                local_rows: list[dict] = []
                local_vectors: list[np.ndarray] = []
                for rank, candidate in enumerate(pool):
                    if rank >= probe_count and time.perf_counter() - start > args.max_seconds:
                        break
                    trial = selected.clone()
                    trial[-1] = candidate
                    loss, margin = evaluate(model, hook, x, int(label), trial,
                                            center_id, seed, device)
                    record = {
                        "sample_pos": sample_pos, "label": int(label),
                        "stage": args.stage, "center": center_id,
                        "candidate": candidate, "replaced": int(selected[-1]),
                        "base_loss": base_loss, "loss": loss,
                        "loss_gain": base_loss - loss,
                        "margin_gain": margin - base_margin,
                    }
                    for key in ("l2", "cosine", "novelty", "dt", "polarity"):
                        record[key] = float(feat[key][candidate].item())
                    local_rows.append(record)
                    local_vectors.append((features[candidate] - centers[center_id]).cpu().numpy())
                sync(device)
                elapsed = time.perf_counter() - start
                valid_count = int(feat["valid"].sum().item()) - len(selected)
                swap = greedy_swaps(model, hook, x, int(label), selected,
                                    center_id, seed, device, local_rows,
                                    args.swap_top, args.swap_rounds, base_loss)
                summary = {
                    "sample_pos": sample_pos, "stage": args.stage,
                    "center": center_id, "valid_candidates": valid_count,
                    "feature_scan_seconds": feature_seconds,
                    "forward_probes": len(local_rows), "forward_seconds": elapsed,
                    "estimated_full_seconds": elapsed / max(len(local_rows), 1) * valid_count,
                    "gpu_peak_allocated_gib": (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        if device.type == "cuda" else None),
                    "gpu_peak_reserved_gib": (
                        torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                        if device.type == "cuda" else None),
                    "base_loss": base_loss,
                    "best_single_swap_gain": max((r["loss_gain"] for r in local_rows), default=None),
                    "pca": pca_summary(local_rows, local_vectors),
                    "greedy": swap,
                    "fixed_rules": rule_results,
                }
                rows.extend(local_rows)
                summaries.append(summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                hook.center = None
                hook.replacement = None
    finally:
        handle.remove()
        hook.remove()

    if rows:
        with (out_dir / "candidate_contributions.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "checkpoint": args.log_dir, "weight": "ema" if args.use_ema else "raw",
        "split": "train", "stage": args.stage, "seed": args.seed,
        "candidate_rule": "l2_near+xy_novel+time_far+random",
        "rule_selection": "exact_global_topk; baseline uses chunked selection",
        "candidate_pools": {
            "fixed_rules": "all_nonlocal_points",
            "single_swap_probe": "all_nonlocal_points_except_baseline_16",
        },
        "cosine_sanity_check": (
            "cosine_near 与 checkpoint baseline 应接近；分块/精确打分、数值并列可导致索引不同，"
            "逐中心查看 overlap_with_baseline，不要求完全相等。"),
        "random_rule": "independent SeedSequence(seed, sample_pos, center, 1)",
        "gpu_peak_scope": "each center: full feature scan, fixed-rule forwards, single-swap forwards, greedy verification",
        "replace_slot": "last_of_checkpoint_selector",
        "summaries": summaries,
        "note": (
            "单点替换只是贡献探针，不是最优16点或重训练准确率；train CE 属拟合态，"
            "不能据此声称泛化增益。四规则彼此同池，单点探针池另排除了 baseline 16 点。"),
    }
    (out_dir / "probe_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}")


def parse_args() -> argparse.Namespace:
    """解析诊断参数，默认只运行一个训练样本的两个中心。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", default="secnet_pi2_g16_cosine_baidu")
    parser.add_argument("--ckpt", choices=("best", "last"), default="best")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--num_point", type=int, default=10240)
    parser.add_argument("--num_category", type=int, default=101)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--sample_ids", nargs="+", type=int)
    parser.add_argument("--centers_per_sample", type=int, default=2)
    parser.add_argument("--pool_size", type=int, default=64)
    parser.add_argument("--scan_all", action="store_true",
                        help="用全候选作探针池；仍受 max_seconds 限制")
    parser.add_argument("--probe_count", type=int, default=8)
    parser.add_argument("--max_seconds", type=float, default=120)
    parser.add_argument("--swap_top", type=int, default=2)
    parser.add_argument("--swap_rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="~/zhaojin/logs_run/selector_utility_probe")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
