"""比较归因高/低贡献点的空间、时间和模型特征，只输出数值。

优先比较同中心、同一贪心轮次的高低贡献候选。只有 NPZ 时可比较
选中集合的贡献两端，但明确标注轮次混杂。通过重放并核对 NPZ
中的坐标/特征来校验 checkpoint、stage 和样本映射，不重新找点。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "x": ("空间", "x 坐标"), "y": ("空间", "y 坐标"),
    "dx": ("空间", "相对中心 x"), "dy": ("空间", "相对中心 y"),
    "xy_distance": ("空间", "到中心 XY 距离"),
    "local_novelty": ("空间", "到局部邻居的最小 XY 距离"),
    "quadrant_1": ("空间", "相对中心第一象限比例"),
    "quadrant_2": ("空间", "相对中心第二象限比例"),
    "quadrant_3": ("空间", "相对中心第三象限比例"),
    "quadrant_4": ("空间", "相对中心第四象限比例"),
    "time_fraction": ("时间", "stage 时间范围中的相对位置"),
    "timestamp": ("时间", "还原坐标单位后的时间戳"),
    "dt": ("时间", "相对中心时间差"),
    "abs_dt": ("时间", "相对中心绝对时间差"),
    "center_cosine": ("特征", "与中心的特征余弦"),
    "neighbor_cosine": ("特征", "与局部邻居的平均特征余弦"),
    "neighbor_cosine_std": ("特征", "与邻居余弦的标准差"),
    "feature_norm": ("特征", "中间特征模长"),
    "direction_variance": ("特征", "组内单位特征方向总方差"),
}


def read_rows(path: Path) -> list[dict]:
    """读取归因 CSV，只转换配对所需的列。"""
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key in ("sample", "center", "rank", "index"):
            row[key] = int(row[key])
        row["gain"] = float(row["gain"])
    return rows


def discover_files(directory: Path) -> dict[tuple[int, int], Path]:
    """只识别逐中心归因 NPZ，不把汇总文件算成样本。"""
    files = {}
    for path in directory.glob("sample*_center*.npz"):
        match = re.fullmatch(r"sample(\d+)_center(\d+)\.npz", path.name)
        if match:
            files[tuple(map(int, match.groups()))] = path
    if not files:
        raise ValueError("未找到 sample<编号>_center<编号>.npz")
    return files


def pairs_from_csv(directory: Path, top_k: int) -> list[dict]:
    """选每中心贡献最高的正增益点，与同轮最差不同候选配对。"""
    groups, lows = defaultdict(list), defaultdict(list)
    for row in read_rows(directory / "selected_points.csv"):
        groups[row["sample"], row["center"]].append(row)
    for row in read_rows(directory / "low_contribution_points.csv"):
        lows[row["sample"], row["center"], row["rank"]].append(row)
    pairs = []
    for (sample, center), rows in sorted(groups.items()):
        high = sorted((r for r in rows if r["gain"] > 0),
                      key=lambda r: r["gain"], reverse=True)[:top_k]
        for row in high:
            candidates = [r for r in lows[sample, center, row["rank"]]
                          if r["index"] != row["index"]]
            if not candidates:
                continue
            low = min(candidates, key=lambda r: r["gain"])
            pairs.append({"sample": sample, "center": center,
                          "rank": row["rank"], "high_index": row["index"],
                          "low_index": low["index"], "high_gain": row["gain"],
                          "low_gain": low["gain"]})
    return pairs


def pairs_from_npz(files: dict, top_k: int) -> list[dict]:
    """在选中 16 点内取贡献两端；此比较包含贪心轮次差异。"""
    pairs = []
    for (sample, center), path in sorted(files.items()):
        with np.load(path, allow_pickle=False) as saved:
            ids, gains = saved["indices"], saved["gains"]
        order = np.argsort(gains, kind="stable")
        high = [int(i) for i in order[::-1] if gains[i] > 0][:top_k]
        low = [int(i) for i in order if i not in high][:len(high)]
        for upper, lower in zip(high, low):
            pairs.append({"sample": sample, "center": center,
                          "rank": upper + 1, "low_rank": lower + 1,
                          "high_index": int(ids[upper]),
                          "low_index": int(ids[lower]),
                          "high_gain": float(gains[upper]),
                          "low_gain": float(gains[lower])})
    return pairs


def source_split(directory: Path, explicit: str | None) -> str:
    """优先使用归因汇总的来源，不允许把 train 索引解释成 test。"""
    summary = directory / "summary.json"
    recorded = None
    if summary.exists():
        source = str(json.loads(summary.read_text(encoding="utf-8"))
                     .get("source", ""))
        recorded = next((name for name in ("train", "test")
                         if f"{name}.h5" in source), None)
    if recorded and explicit and recorded != explicit:
        raise ValueError(f"归因来自 {recorded}.h5，不能配对 {explicit}.h5")
    if not (recorded or explicit):
        raise ValueError("来源未记录，请按生成记录指定 --source_split train 或 test")
    return recorded or explicit


def normalized(matrix: np.ndarray) -> np.ndarray:
    """按最后一维归一化，零向量保留为零。"""
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-8)


def point_metrics(index: int, center: int, capture: dict) -> dict[str, float]:
    """计算候选点相对 FPS 中心和局部 24 点的几何、时间和特征。"""
    xyz, features = capture["xyz"], capture["features"]
    local_ids = capture["local"][center]
    origin = capture["fps"][center]
    delta = xyz[index] - xyz[origin]
    dx, dy = float(delta[1]), float(delta[2])
    candidate = normalized(features[index])
    similarity = normalized(features[local_ids]) @ candidate
    times = xyz[:, 0]
    start, end = capture.get("time_range", (float(times.min()), float(times.max())))
    span = max(float(end - start), 1e-8)
    return {
        "x": float(xyz[index, 1]), "y": float(xyz[index, 2]), "dx": dx, "dy": dy,
        "xy_distance": float(np.linalg.norm(delta[1:3])),
        "local_novelty": float(np.linalg.norm(
            xyz[local_ids, 1:3] - xyz[index, 1:3], axis=-1).min()),
        "quadrant_1": float(dx >= 0 and dy >= 0),
        "quadrant_2": float(dx < 0 and dy >= 0),
        "quadrant_3": float(dx < 0 and dy < 0),
        "quadrant_4": float(dx >= 0 and dy < 0),
        "time_fraction": float((xyz[index, 0] - start) / span),
        "timestamp": float(xyz[index, 0]),
        "dt": float(delta[0]), "abs_dt": float(abs(delta[0])),
        "center_cosine": float(candidate @ normalized(features[origin])),
        "neighbor_cosine": float(similarity.mean()),
        "neighbor_cosine_std": float(similarity.std()),
        "feature_norm": float(np.linalg.norm(features[index])),
    }


def verify_artifact(path: Path, capture: dict) -> dict:
    """用原 NPZ 坐标和特征核对 stage、checkpoint、样本与 FPS 重放。"""
    differences = {}
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["indices"].astype(np.int64)
        for key, actual in (("xyz", capture.get("scaled_xyz", capture["xyz"])),
                            ("feature", capture["features"])):
            stored = saved[key].astype(np.float32)
            if ids.min() < 0 or ids.max() >= len(actual):
                raise ValueError(f"{path.name}: stage 点索引越界")
            replay = actual[ids]
            if stored.shape != replay.shape or not np.allclose(
                    stored, replay, rtol=0.005, atol=0.005):
                raise ValueError(f"{path.name}: {key} 与重放不匹配；检查来源、"
                                 "sample_id_map、stage、checkpoint、EMA 和 seed")
            differences[key] = float(np.max(np.abs(stored - replay)))
    return differences


def verify_pairs(files: dict, pairs: list[dict]) -> None:
    """核对 CSV 中的高贡献点确实对应 NPZ 的该轮选点及增益。"""
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["sample"], pair["center"]].append(pair)
    for key, group in grouped.items():
        with np.load(files[key], allow_pickle=False) as saved:
            for pair in group:
                rank = pair["rank"] - 1
                if (int(saved["indices"][rank]) != pair["high_index"] or
                    not np.isclose(saved["gains"][rank], pair["high_gain"],
                                   rtol=1e-4, atol=1e-6)):
                    raise ValueError(f"{files[key].name}: CSV 与 NPZ 的选点记录不一致")


def enrich_pairs(pairs: list[dict], capture: dict) -> list[dict]:
    """补齐高低贡献点的统计指标，并计算每中心组内方向方差。"""
    groups = defaultdict(list)
    for pair in pairs:
        if not 0 <= pair["center"] < len(capture["local"]):
            raise ValueError("CSV 中心编号超出重放 stage 范围")
        if any(not 0 <= pair[f"{name}_index"] < len(capture["features"])
               for name in ("high", "low")):
            raise ValueError("CSV 候选点索引超出重放 stage 范围")
        groups[pair["center"]].append(pair)
    output = []
    for center, group in groups.items():
        spread = {}
        for name in ("high", "low"):
            indices = [p[f"{name}_index"] for p in group]
            vectors = normalized(capture["features"][indices])
            spread[name] = float(np.var(vectors, axis=0).sum())
        for pair in group:
            enriched = dict(pair)
            for name in ("high", "low"):
                values = point_metrics(pair[f"{name}_index"], center, capture)
                values["direction_variance"] = spread[name]
                enriched[name] = values
            output.append(enriched)
    return output


def paired_samples(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """先按中心、再按样本平均，避免把同图像里的点当成独立重复。"""
    centers = defaultdict(list)
    for record in records:
        centers[record["sample"], record["center"]].append(
            [record["high"][key], record["low"][key]])
    samples = defaultdict(list)
    for (sample, _), values in centers.items():
        samples[sample].append(np.mean(values, axis=0))
    matrix = np.array([np.mean(samples[sample], axis=0)
                       for sample in sorted(samples)])
    return matrix[:, 0], matrix[:, 1]


def paired_test(high: np.ndarray, low: np.ndarray, seed: int = 42) -> dict:
    """按样本做配对符号翻转检验和样本 bootstrap 置信区间。"""
    difference = high - low
    count = len(difference)
    if count < 3:
        return {"p": None, "mean_difference": float(difference.mean()),
                "ci95": None, "effect_dz": None, "status": "fewer_than_3_samples"}
    rng = np.random.default_rng(seed)
    observed = abs(float(difference.mean()))
    if count <= 15:
        signs = np.array(list(itertools.product((-1, 1), repeat=count)))
        p_value = float(np.mean(np.abs((signs * difference).mean(1)) >= observed - 1e-12))
    else:
        signs = rng.choice((-1, 1), size=(10000, count))
        extreme = np.sum(np.abs((signs * difference).mean(1)) >= observed - 1e-12)
        p_value = float((extreme + 1) / 10001)
    bootstrap = difference[rng.integers(count, size=(5000, count))].mean(1)
    deviation = float(difference.std(ddof=1))
    return {"p": p_value, "mean_difference": float(difference.mean()),
            "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "effect_dz": float(difference.mean() / deviation) if deviation > 0 else None,
            "status": "sample_paired_sign_flip"}


def adjust_fdr(results: dict) -> None:
    """对全部预定义指标的 p 值做 Benjamini-Hochberg 多重比较校正。"""
    valid = sorted((value["p"], key) for key, value in results.items()
                   if value["p"] is not None)
    previous = 1.0
    for rank in range(len(valid), 0, -1):
        p_value, key = valid[rank - 1]
        previous = min(previous, p_value * len(valid) / rank)
        results[key]["q"] = float(previous)
        results[key]["significant"] = previous < .05


def summarize(records: list[dict]) -> dict:
    """给出两组描述统计、按样本配对检验及 FDR 结果。"""
    results = {}
    for key, (category, label) in METRICS.items():
        high, low = paired_samples(records, key)
        result = {"category": category, "label": label,
                  "n_samples": len(high), "q": None, "significant": False}
        for name in ("high", "low"):
            points = np.array([r[name][key] for r in records])
            sample_values = high if name == "high" else low
            result[name] = {"mean": float(sample_values.mean()),
                            "sample_mean_variance": float(sample_values.var()),
                            "point_mean": float(points.mean()),
                            "point_variance": float(points.var()),
                            "point_std": float(points.std())}
        result.update(paired_test(high, low))
        results[key] = result
    adjust_fdr(results)
    return results


def dependency_status() -> dict[str, bool]:
    """兼容 tools 平铺脚本及 models/modules.py 的实际部署布局。"""
    root = Path(__file__).resolve().parent
    status = {name: (root / name).is_file() for name in
              ("point_attribution_analysis.py", "selector_utility_probe.py", "extract_preds.py")}
    status["modules.py"] = any(path.is_file() for path in
                               (root / "modules.py", root.parent / "models" / "modules.py"))
    return status


def preflight() -> None:
    """在服务器运行时先检查归因脚本及模型模块是否已部署。"""
    missing = [name for name, exists in dependency_status().items() if not exists]
    if missing:
        raise FileNotFoundError(f"同目录缺少依赖：{', '.join(missing)}")


def inspect_inputs(directory: Path, files: dict) -> dict:
    """打印一个 NPZ 的真实字段及服务器依赖文件是否存在。"""
    first = next(iter(sorted(files.values())))
    with np.load(first, allow_pickle=False) as saved:
        fields = {key: {"shape": list(saved[key].shape), "dtype": str(saved[key].dtype)}
                  for key in saved.files}
    result = {"npz_count": len(files), "example_file": str(first), "fields": fields,
              "dependencies": dependency_status(),
              "csv": {name: (directory / name).is_file() for name in
                  ("selected_points.csv", "low_contribution_points.csv")}}
    summary = directory / "summary.json"
    result["recorded_source"] = (json.loads(summary.read_text(encoding="utf-8"))
                                  .get("source") if summary.exists() else None)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def coordinate_scale(model, stage: int) -> np.ndarray:
    """求各 stage 累积缩放，恢复原输入单位中的邻域平均坐标。"""
    scale = np.ones(4, dtype=np.float64)
    for grouper in model.local_grouper_list[:stage + 1]:
        scale *= grouper.fps_parameter.detach().cpu().numpy().reshape(4)
    if np.any(np.abs(scale) < 1e-8):
        raise ValueError("累计坐标缩放接近零，不能可靠还原时间/空间单位")
    return scale


def capture_sample(model, selector, intervention, raw, seed, device, stage) -> dict:
    """只做一次模型前向，抓取指定 stage 的全部候选及真实 FPS 中心。"""
    import torch
    from selector_utility_probe import predict
    raw = np.asarray(raw).reshape(-1, 4)
    x = torch.from_numpy(raw).float().T.unsqueeze(0).to(device)
    predict(model, x, seed, device)
    convert = lambda value: value.detach().cpu().numpy()
    scaled_xyz = convert(intervention.input_xyz[0])
    capture = {"xyz": scaled_xyz / coordinate_scale(model, stage),
               "scaled_xyz": scaled_xyz,
               "time_range": (float(raw[:, 0].min()), float(raw[:, 0].max())),
               "features": convert(selector.all_feat[0]),
               "local": convert(selector.local_idx[0]),
               "fps": convert(intervention.fps_idx[0])}
    actual = capture["features"][capture["fps"]]
    if not np.allclose(actual, convert(selector.center_feat[0]), rtol=1e-5, atol=1e-5):
        raise ValueError("FPS 重放与 selector 中心不一致")
    return capture


def replay_records(args, files: dict, pairs: list[dict], split: str):
    """按样本重放模型，核对原始归因后提取两组点的统计。"""
    import h5py
    import torch
    from point_attribution_analysis import SubsetIntervention
    from selector_utility_probe import SelectorIntervention, load_model
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args, device)
    grouper = model.local_grouper_list[args.stage]
    selector = SelectorIntervention(grouper.remote_selector)
    selector.install()
    intervention = SubsetIntervention(grouper, selector)
    handle = grouper.register_forward_pre_hook(intervention.capture_input)
    mapping = (json.loads(Path(args.sample_id_map).read_text(encoding="utf-8"))
               if args.sample_id_map else {})
    by_sample = defaultdict(list)
    for pair in pairs:
        by_sample[pair["sample"]].append(pair)
    records, checks = [], []
    try:
        with h5py.File(Path(args.data_path).expanduser() / f"{split}.h5", "r") as data:
            for sample, sample_pairs in sorted(by_sample.items()):
                source_id = int(mapping.get(str(sample), sample))
                if source_id < 0 or source_id >= len(data["data"]):
                    raise ValueError("样本映射超出 H5 范围")
                capture = capture_sample(model, selector, intervention,
                    data["data"][source_id], args.seed + sample, device, args.stage)
                for center in sorted({p["center"] for p in sample_pairs}):
                    difference = verify_artifact(files[sample, center], capture)
                    checks.append({"sample": sample, "source_sample_id": source_id,
                                   "center": center, "max_abs_difference": difference})
                records.extend(enrich_pairs(sample_pairs, capture))
                print(f"已核验样本 {sample} → {split}[{source_id}]", flush=True)
    finally:
        handle.remove()
        selector.remove()
    return records, checks


def conclusion(results: dict) -> str:
    """只根据校正后统计给结论，不把相关性表述为选点的因果原因。"""
    significant = [r for r in results.values() if r["significant"]]
    if not significant:
        return "尚未发现高低贡献点经多重比较校正后的稳定特征差异。"
    selected = []
    for category in ("空间", "时间", "特征"):
        group = [r for r in significant if r["category"] == category]
        if group:
            best = min(group, key=lambda r: r["q"])
            direction = "更高" if best["mean_difference"] > 0 else "更低"
            selected.append(best["label"] + direction)
    return ("当前归因中，高贡献点表现为" + "、".join(selected) +
            "（FDR q<0.05）；这些关联仍需选点消融验证。")


def run(args: argparse.Namespace) -> None:
    """分析既有归因文件，不重新搜索候选，不修改训练代码。"""
    directory = Path(args.attribution_dir).expanduser()
    files = discover_files(directory)
    inspection = inspect_inputs(directory, files)
    if args.inspect_only:
        return
    preflight()
    split = source_split(directory, args.source_split)
    has_csv = all((directory / name).exists() for name in
                  ("selected_points.csv", "low_contribution_points.csv"))
    mode = args.comparison
    if mode == "auto":
        mode = "same_round" if has_csv else "selected_extremes"
    pairs = (pairs_from_csv(directory, args.top_k) if mode == "same_round" else
             pairs_from_npz(files, args.top_k))
    missing = {(p["sample"], p["center"]) for p in pairs} - files.keys()
    if missing or not pairs:
        raise ValueError(f"归因配对为空或缺少中心 NPZ：{sorted(missing)}")
    verify_pairs(files, pairs)
    print(f"来源={split}.h5；比较方式={mode}；有效配对={len(pairs)}", flush=True)
    if mode == "selected_extremes":
        print("仅有选中点两端比较，含贪心轮次混杂，不能称后选点为噪声。", flush=True)
    records, checks = replay_records(args, files, pairs, split)
    results = summarize(records)
    output = {"source_split": split, "checkpoint": args.log_dir,
              "weight": "ema" if args.use_ema else "raw", "stage": args.stage,
              "ckpt": args.ckpt, "seed": args.seed,
              "data_path": str(Path(args.data_path).expanduser().resolve()),
              "comparison": mode, "npz_files": len(files), "pairs": len(records),
              "n_samples": len({r["sample"] for r in records}),
              "n_centers": len({(r["sample"], r["center"]) for r in records}),
              "inspection": inspection,
              "coordinate_layout": "t,x,y,p; cumulative fps scaling undone for metrics",
              "coordinate_meaning": "stage>0 points are neighborhood averages, not raw event rows",
              "inference_unit": "paired sample means after center means",
              "metrics": results, "artifact_checks": checks,
              "conclusion": conclusion(results),
              "limitations": ["低贡献候选不一定无用，结论限于已扫描候选与固定模型",
                              "selected_extremes 比较有贪心轮次混杂；same_round 控制了该混杂",
                              "特征差异是关联证据；标签驱动的选点需另做留出验证"]}
    destination = Path(args.output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
    print(f"统计使用 {output['n_samples']} 个样本、{output['n_centers']} 个中心。")
    print_report(results)
    print(output["conclusion"])
    print(f"保存：{destination}")


def print_report(results: dict) -> None:
    """打印三组均值、方差、配对 p 值和校正后的显著性。"""
    for key, result in results.items():
        high, low = result["high"], result["low"]
        p_value = "样本不足" if result["p"] is None else f"{result['p']:.5g}"
        q_value = "NA" if result["q"] is None else f"{result['q']:.5g}"
        print(f"[{result['category']}] {result['label']}：高贡献均值="
              f"{high['point_mean']:.5g}，点方差={high['point_variance']:.5g}；"
              f"低贡献均值={low['point_mean']:.5g}，点方差={low['point_variance']:.5g}；"
              f"p={p_value}，q={q_value}，"
              f"{'显著' if result['significant'] else '未达显著'}")


def parse_args() -> argparse.Namespace:
    """要求显式指定生成归因时的 checkpoint 与 stage，避免跨实验混用。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution_dir", default="~/zhaojin/logs_run/point_attribution")
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--source_split", choices=["train", "test"])
    parser.add_argument("--log_dir")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2])
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--num_category", type=int, default=101)
    parser.add_argument("--sample_id_map", help="JSON 对象：归因样本编号到原 H5 行号")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--comparison", choices=["auto", "same_round", "selected_extremes"],
                        default="auto")
    parser.add_argument("--inspect_only", action="store_true",
                        help="只检查一个 NPZ 的字段及服务器依赖，不加载模型")
    parser.add_argument("--device", default="cpu", help="cpu/auto/cuda:0；默认 CPU，避免与训练抢卡")
    parser.add_argument("--output", default="~/zhaojin/logs_run/point_importance_analysis.json")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 8:
        parser.error("top_k 必须在 [1,8] 内")
    if not args.inspect_only and (args.log_dir is None or args.stage is None):
        parser.error("完整分析须指定生成归因时的 --log_dir 和 --stage")
    return args


if __name__ == "__main__":
    run(parse_args())
