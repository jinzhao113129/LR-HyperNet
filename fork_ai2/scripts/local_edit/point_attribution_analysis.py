"""逐点测试每个 FPS 中心的分类贡献，并解释选中 16 点的几何关系。

空集定义为该中心远程分支的零向量；非空集合仍经 checkpoint 中的
hyper_agg 聚合。默认用混合短名单限制前向次数；--candidate_cap 0 可
显式枚举全部 N 个点，但 16 轮的计算量极大。只读取 train.h5。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn.functional as F

from selector_utility_probe import (ROOT, SelectorIntervention, load_model,
                                    predict, scores_for_all, train_examples)
from modules import furthest_point_sample


class SubsetIntervention:
    """仅替换指定中心的远程聚合结果，其余前向保持 checkpoint 原样。"""

    def __init__(self, grouper: torch.nn.Module, selector: SelectorIntervention):
        """保存聚合器和待注入的中心/点集。"""
        self.grouper = grouper
        self.selector = selector
        self.center: int | None = None
        self.selected: list[int] = []
        self.input_xyz: torch.Tensor | None = None
        self.fps_idx: torch.Tensor | None = None

    def capture_input(self, module, inputs) -> None:
        """缓存本 stage 坐标，并重放 FPS 索引以定位真实中心。"""
        xytp = inputs[0].detach()
        self.input_xyz = module.fps_parameter * xytp
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(xytp.device) if xytp.is_cuda else None
        idx = furthest_point_sample(self.input_xyz.contiguous()[:, :, :3],
                                    module.groups).long()
        self.fps_idx = idx.sort(dim=1).values
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, xytp.device)

    def hook(self, _module, _inputs, output):
        """把一个中心的 remote feature 替换为空集或选中集合的聚合。"""
        if self.center is None:
            return output
        coords, grouped = output
        width = self.selector.all_feat.shape[-1]
        if self.selected:
            indices = torch.as_tensor(self.selected, device=grouped.device)
            points = self.selector.all_feat[:, indices].unsqueeze(1)
            remote = self.grouper.hyper_agg(points)[:, 0]
        else:
            remote = torch.zeros((grouped.shape[0], width), device=grouped.device,
                                 dtype=grouped.dtype)
        changed = grouped.clone()
        changed[:, self.center, :, -width:] = remote.unsqueeze(1)
        return coords, changed


def loss_and_logit(model: torch.nn.Module, x: torch.Tensor, label: int,
                   seed: int, device: torch.device) -> tuple[float, float]:
    """完整前向得到交叉熵和目标类 logit。"""
    logits = predict(model, x, seed, device)
    loss = F.cross_entropy(logits, torch.tensor([label], device=device)).item()
    return loss, float(logits[0, label].item())


def mixed_shortlist(feat: dict[str, torch.Tensor], cap: int,
                    rng: np.random.Generator, chosen: list[int]) -> list[int]:
    """从近/远特征、空间新颖、时间差、随机五路抽候选。"""
    all_ids = np.arange(len(feat["l2"]))
    available = np.setdiff1d(all_ids, np.array(chosen, dtype=np.int64))
    if cap == 0 or cap >= len(available):
        return available.tolist()
    quota = max(1, (cap + 4) // 5)
    ids = torch.as_tensor(available, device=feat["l2"].device)
    result = []
    for key, descending in (("l2", False), ("l2", True),
                            ("novelty", True), ("dt", True)):
        order = feat[key][ids].topk(min(quota, len(ids)),
                                    largest=descending).indices
        result.extend(ids[order].cpu().tolist())
    result.extend(rng.choice(available, min(quota, len(available)),
                             replace=False).tolist())
    unique = list(dict.fromkeys(result))
    return unique[:cap]


def spatial_direction(center_xyz: np.ndarray, local_xyz: np.ndarray,
                      point_xyz: np.ndarray) -> tuple[float, float]:
    """测点沿局部 XY 第一主方向的投影与垂直偏移。"""
    local_xy = local_xyz[:, 1:3].astype(np.float64)
    local_xy -= local_xy.mean(0)
    _, _, vh = np.linalg.svd(local_xy, full_matrices=False)
    delta = point_xyz[1:3] - center_xyz[1:3]
    parallel = float(delta @ vh[0])
    perpendicular = float(abs(delta @ vh[1]))
    return parallel, perpendicular


def geometry_row(feat: dict[str, torch.Tensor], candidate: int,
                 center_xyz: np.ndarray, local_xyz: np.ndarray,
                 point_xyz: np.ndarray) -> dict:
    """提取一个点的空间、时间、特征与局部 PCA 统计。"""
    along, across = spatial_direction(center_xyz, local_xyz, point_xyz)
    return {
        "xy_distance": float(np.linalg.norm(point_xyz[1:3] - center_xyz[1:3])),
        "time_delta": float(abs(point_xyz[0] - center_xyz[0])),
        "feature_cosine": float(feat["cosine"][candidate]),
        "feature_l2": float(feat["l2"][candidate]),
        "local_xy_novelty": float(feat["novelty"][candidate]),
        "pca_along": along, "pca_across": across,
    }


def search_center(model: torch.nn.Module, intervention: SubsetIntervention,
                  x: torch.Tensor, label: int, center: int, seed: int,
                  feat: dict[str, torch.Tensor], xyz: torch.Tensor,
                  center_xyz: torch.Tensor,
                  local_idx: torch.Tensor, rng: np.random.Generator,
                  args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    """从空集开始逐轮贪心；每个候选均以完整前向计边际损失。"""
    device = x.device
    intervention.center = center
    intervention.selected = []
    current_loss, current_logit = loss_and_logit(model, x, label, seed, device)
    empty_loss = current_loss
    selected: list[int] = []
    rows, bad_rows = [], []
    center_xyz = center_xyz.cpu().numpy()
    local_xyz = xyz[0, local_idx].cpu().numpy()
    for rank in range(1, 17):
        candidates = mixed_shortlist(feat, args.candidate_cap, rng, selected)
        best = None
        scanned = []
        for candidate in candidates:
            intervention.selected = selected + [candidate]
            loss, logit = loss_and_logit(model, x, label, seed, device)
            scanned.append((loss, candidate))
            if best is None or loss < best[0]:
                best = (loss, logit, candidate)
        if best is None:
            raise RuntimeError("没有可用候选点")
        loss, logit, candidate = best
        intervention.selected = selected + [candidate]
        verify_loss, _ = loss_and_logit(model, x, label, seed, device)
        if abs(loss - verify_loss) > 1e-4:
            raise RuntimeError("相同点集重复前向损失不一致")
        point_xyz = xyz[0, candidate].cpu().numpy()
        row = geometry_row(feat, candidate, center_xyz, local_xyz, point_xyz)
        row.update({"rank": rank, "index": candidate,
                    "gain": current_loss - loss,
                    "target_logit_gain": logit - current_logit,
                    "loss_after": loss, "scanned": len(candidates)})
        rows.append(row)
        for bad_loss, bad_idx in sorted(scanned, reverse=True)[:3]:
            bad = geometry_row(feat, bad_idx, center_xyz, local_xyz,
                               xyz[0, bad_idx].cpu().numpy())
            bad.update({"rank": rank, "index": bad_idx,
                        "gain": current_loss - bad_loss})
            bad_rows.append(bad)
        selected.append(candidate)
        current_loss, current_logit = loss, logit
    return rows, bad_rows, {"empty_loss": empty_loss, "final_loss": current_loss,
                            "total_gain": empty_loss - current_loss}


def save_sample(out: Path, sample: int, center: int,
                rows: list[dict], stats: dict, feature: torch.Tensor,
                xyz: torch.Tensor) -> None:
    """保存按边际贡献顺序排列的索引、坐标、特征和单点统计。"""
    ids = torch.as_tensor([r["index"] for r in rows], device=feature.device)
    arr = lambda tensor: tensor.detach().cpu().numpy().astype(np.float16)
    np.savez_compressed(out / f"sample{sample}_center{center}.npz",
                        indices=ids.cpu().numpy(), xyz=arr(xyz[ids]),
                        feature=arr(feature[ids]),
                        gains=np.array([r["gain"] for r in rows], dtype=np.float32),
                        logit_gains=np.array([r["target_logit_gain"] for r in rows],
                                              dtype=np.float32),
                        empty_loss=np.float32(stats["empty_loss"]),
                        final_loss=np.float32(stats["final_loss"]))


def save_summary(out: Path, rows: list[dict], bad_rows: list[dict],
                 overview: list[dict]) -> None:
    """写 CSV、汇总 JSON 与 16 步平均边际贡献曲线。"""
    if not rows:
        raise RuntimeError("没有完成任何中心的归因")
    with (out / "selected_points.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "low_contribution_points.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bad_rows[0]))
        writer.writeheader()
        writer.writerows(bad_rows)
    summaries = {}
    for key in ("xy_distance", "time_delta", "feature_cosine", "feature_l2",
                "local_xy_novelty", "pca_along", "pca_across"):
        summaries[key] = {}
        for name, source in (("selected", rows), ("low_contribution", bad_rows)):
            values = np.array([r[key] for r in source])
            summaries[key][name] = {
                "mean": float(values.mean()), "std": float(values.std()),
                "q10_q50_q90": np.quantile(values, [.1, .5, .9]).tolist()}
    report = {"source": "train.h5 only", "search": overview,
              "selected_point_statistics": summaries,
              "empty_definition": "zero remote branch at one center",
              "warning": "若 candidate_cap>0 则非全点穷举；训练集归因不能证明泛化。"}
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    means = [np.mean([r["gain"] for r in rows if r["rank"] == i])
             for i in range(1, 17)]
    if plt is not None:
        plt.plot(range(1, 17), means, marker="o")
        plt.xlabel("Selected point rank")
        plt.ylabel("Mean marginal CE decrease")
        plt.tight_layout()
        plt.savefig(out / "marginal_gain.png", dpi=160)
        plt.close()
    else:
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        lo, hi = min(means), max(means)
        span = max(hi - lo, 1e-8)
        points = [(40 + i * 38, int(320 - 280 * (v - lo) / span))
                  for i, v in enumerate(means)]
        draw.line(points, fill="blue", width=3)
        draw.line([(40, 20), (40, 320), (620, 320)], fill="black", width=2)
        image.save(out / "marginal_gain.png")


def analyze_sample(model, selector, intervention, raw, label, sample,
                   args, rng, out) -> tuple[list[dict], list[dict], list[dict]]:
    """对一个训练样本的指定 FPS 中心执行归因。"""
    x = torch.from_numpy(raw).float().permute(1, 0).unsqueeze(0).to(next(model.parameters()).device)
    intervention.center = None
    predict(model, x, args.seed + sample, x.device)
    features = selector.all_feat[0].clone()
    locals_ = selector.local_idx[0].clone()
    centers = selector.center_feat[0].clone()
    xyz = intervention.input_xyz[0].clone()
    fps_idx = intervention.fps_idx[0].clone()
    chosen = rng.choice(len(centers), min(args.centers_per_sample,
                                          len(centers)), replace=False)
    rows, bad_rows, overview = [], [], []
    for center in chosen:
        center = int(center)
        center_xyz = xyz[fps_idx[center]]
        if not torch.allclose(features[fps_idx[center]], centers[center]):
            raise RuntimeError("FPS 中心重放不一致")
        feat = scores_for_all(centers[center], features,
                              locals_[center], selector.baseline_idx[0, center],
                              center_xyz[:3], xyz[:, :3], xyz[:, 3])
        started = time.monotonic()
        point_rows, poor_rows, stat = search_center(
            model, intervention, x, int(label), center, args.seed + sample,
            feat, xyz.unsqueeze(0), center_xyz, locals_[center], rng, args)
        stat.update({"sample": sample, "center": center,
                     "seconds": time.monotonic() - started,
                     "candidate_cap": args.candidate_cap})
        for row in point_rows + poor_rows:
            row.update({"sample": sample, "center": center, "label": int(label)})
        rows.extend(point_rows)
        bad_rows.extend(poor_rows)
        overview.append(stat)
        save_sample(out, sample, center, point_rows, stat, features, xyz)
        print(json.dumps(stat, ensure_ascii=False), flush=True)
    return rows, bad_rows, overview


def run(args: argparse.Namespace) -> None:
    """装载固定 G16 checkpoint，逐样本逐中心执行诊断。"""
    planned = args.max_samples * args.centers_per_sample * 16 * (
        args.num_point if args.candidate_cap == 0 else args.candidate_cap)
    if planned > args.max_planned_forwards and not args.allow_large_run:
        raise ValueError(
            f"计划约 {planned:,} 次完整前向；先缩小 --max_samples/"
            "--centers_per_sample/--candidate_cap 做速度实测。"
            "确认可承受后显式加 --allow_large_run。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    data, labels = train_examples(args)
    grouper = model.local_grouper_list[args.stage]
    if grouper.remote_selector.g16_treatment in {"random", "spatial_fixed"}:
        raise ValueError("随机固定规则的私有 RNG 未重放；请选择确定性 G16 checkpoint")
    if grouper.use_remote_gate or grouper.use_global_anchor or grouper.remote_kda_enc:
        raise ValueError("本脚本暂不支持 remote_gate/global_anchor/remote_kda_enc")
    selector = SelectorIntervention(grouper.remote_selector)
    selector.install()
    intervention = SubsetIntervention(grouper, selector)
    pre_handle = grouper.register_forward_pre_hook(intervention.capture_input)
    handle = grouper.register_forward_hook(intervention.hook)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows, bad_rows, overview = [], [], []
    try:
        for sample, (raw, label) in enumerate(zip(data, labels)):
            good, poor, stats = analyze_sample(model, selector, intervention,
                                                raw, label, sample, args, rng, out)
            rows.extend(good)
            bad_rows.extend(poor)
            overview.extend(stats)
    finally:
        pre_handle.remove()
        handle.remove()
        selector.remove()
    save_summary(out, rows, bad_rows, overview)


def parse_args() -> argparse.Namespace:
    """默认 16 样本×64 中心；首次实跑应显式缩小规模。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", default="secnet_pi2_g16_cosine_baidu")
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--num_point", type=int, default=10240)
    parser.add_argument("--num_category", type=int, default=101)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--centers_per_sample", type=int, default=64)
    parser.add_argument("--candidate_cap", type=int, default=64)
    parser.add_argument("--max_planned_forwards", type=int, default=5000)
    parser.add_argument("--allow_large_run", action="store_true")
    parser.add_argument("--sample_ids", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="~/zhaojin/logs_run/point_attribution")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
