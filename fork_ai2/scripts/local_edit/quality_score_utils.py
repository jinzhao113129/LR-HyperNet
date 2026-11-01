"""综合分验证的纯数值函数：候选、五项信号、数量规则和配对统计。"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


SIGNALS = ("center_xy_distance", "local_xy_novelty", "one_minus_center_cosine",
           "one_minus_neighbor_cosine", "feature_norm")
METHODS = ("cosine16", "quality16", "quality4", "adaptive")


def unit(matrix: np.ndarray) -> np.ndarray:
    """沿特征轴归一化，零向量保留为零。"""
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-8)


def sample_pool(size: int, local: np.ndarray, count: int,
                seed: list[int]) -> np.ndarray:
    """只排除局部点，无放回均匀抽样；不按分数或标签预筛。"""
    valid = np.setdiff1d(np.arange(size), np.unique(local))
    if count < 16 or len(valid) < count:
        raise ValueError("同池四规则比较要求至少 16 个候选，且候选数不超过非局部点数")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return np.sort(rng.choice(valid, count, replace=False))


def quality_signals(features: np.ndarray, xyz: np.ndarray, origin: int,
                    local: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """按原输入坐标单位计算五项信号；邻居余弦为逐邻居余弦的均值。"""
    points = features[pool].astype(np.float64)
    normalized = unit(points)
    local_direction = unit(features[local].astype(np.float64)).mean(axis=0)
    local_dist = xyz[pool, None, 1:3] - xyz[local][None, :, 1:3]
    values = np.stack([
        np.linalg.norm(xyz[pool, 1:3] - xyz[origin, 1:3], axis=-1),
        np.linalg.norm(local_dist, axis=-1).min(axis=1),
        1 - normalized @ unit(features[origin].astype(np.float64)),
        1 - normalized @ local_direction,
        np.linalg.norm(points, axis=-1),
    ], axis=-1)
    if not np.isfinite(values).all():
        raise ValueError("综合分信号出现非有限值")
    return values


def quality_score(signals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """候选池内逐指标 z-score 后加权，常量指标贡献为零。"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (5,) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("weights 必须为五个有限非负数")
    deviation = signals.std(axis=0)
    scaled = (signals - signals.mean(axis=0)) / np.maximum(deviation, 1e-3)
    return scaled @ weights


def adaptive_indices(scores: np.ndarray, minimum: int, maximum: int,
                     ratio: float, absolute: float) -> np.ndarray:
    """按降序截断；至少选 minimum；没有分数差异时只选 minimum。"""
    if not 1 <= minimum <= maximum <= min(16, len(scores)):
        raise ValueError("选点数须满足 1 <= min_g <= max_g <= min(16,M)")
    if not 0 <= ratio <= 1 or not np.isfinite(absolute):
        raise ValueError("stop_ratio 须在 [0,1] 内，stop_abs 须有限")
    if not np.isfinite(scores).all():
        raise ValueError("质量分必须有限")
    order = np.argsort(-scores, kind="stable")
    count = minimum
    if np.ptp(scores) <= 1e-8:
        return order[:count]
    while count < maximum:
        following = scores[order[count]]
        if following < absolute or following < scores[order[:count]].mean() * ratio:
            break
        count += 1
    return order[:count]


def rule_sets(pool: np.ndarray, signals: np.ndarray, scores: np.ndarray,
              minimum: int, maximum: int, ratio: float, absolute: float) -> dict:
    """四规则共享完全相同的候选池；索引排序不改变聚合点集。"""
    cosine = np.argsort(signals[:, 2], kind="stable")
    quality = np.argsort(-scores, kind="stable")
    adaptive = adaptive_indices(scores, minimum, maximum, ratio, absolute)
    return {"cosine16": np.sort(pool[cosine[:16]]).tolist(),
            "quality16": np.sort(pool[quality[:16]]).tolist(),
            "quality4": np.sort(pool[quality[:4]]).tolist(),
            "adaptive": np.sort(pool[adaptive]).tolist()}


def heldout_ids(total: int, discovery: list[int], requested: list[int] | None,
                count: int, seed: int = 42) -> list[int]:
    """从训练集选出未参与五项特征发现的样本，重叠直接报错。"""
    if count < 1 or not discovery or len(set(discovery)) != len(discovery):
        raise ValueError("需要非空、不重复的发现样本清单和正的 max_samples")
    if min(discovery) < 0 or max(discovery) >= total:
        raise ValueError("发现样本编号超出 train.h5 范围")
    excluded = set(discovery)
    available = [i for i in range(total) if i not in excluded]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 891]))
    ids = (requested[:count] if requested is not None else
           rng.choice(available, min(count, len(available)), replace=False).tolist())
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= total:
        raise ValueError("验证样本编号为空、重复或越界")
    if set(ids) & excluded:
        raise ValueError("验证样本与特征发现样本有重叠")
    return ids


def probe_statistics(scores: np.ndarray, gains: np.ndarray) -> dict:
    """在同一个固定 15 点上下文中，测分数与候选替换收益的关系。"""
    valid = np.isfinite(gains)
    score, gain = scores[valid], gains[valid]
    if len(gain) < 4:
        raise ValueError("有效替换候选少于四个")
    rho = None
    if np.ptp(score) > 1e-8 and np.ptp(gain) > 1e-8:
        rho = float(spearmanr(score, gain).statistic)
    order = np.argsort(score, kind="stable")
    quartile = max(1, len(order) // 4)
    return {"n": len(gain), "spearman": rho,
            "top_quartile_gain": float(gain[order[-quartile:]].mean()),
            "bottom_quartile_gain": float(gain[order[:quartile]].mean()),
            "top_bottom_gain": float(gain[order[-quartile:]].mean() -
                                     gain[order[:quartile]].mean()),
            "positive_gain_fraction": float(np.mean(gain > 0))}


def sample_test(values: list[float], seed: int = 42) -> dict:
    """样本层面符号翻转与 bootstrap；不把多个中心当独立图像。"""
    array = np.asarray(values, dtype=np.float64)
    output = {"n_samples": len(array), "mean": float(array.mean()),
              "p": None, "q": None, "ci95": None}
    if len(array) < 3:
        return output
    rng = np.random.default_rng(seed)
    observed = abs(array.mean())
    if len(array) <= 15:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(array))))
        output["p"] = float(np.mean(abs((signs * array).mean(1)) >= observed - 1e-12))
    else:
        signs = rng.choice((-1, 1), size=(10000, len(array)))
        output["p"] = float((np.sum(abs((signs * array).mean(1)) >= observed - 1e-12) + 1) / 10001)
    draws = array[rng.integers(len(array), size=(5000, len(array)))].mean(1)
    output["ci95"] = np.quantile(draws, [.025, .975]).tolist()
    return output


def method_metrics(samples: list[dict]) -> dict:
    """汇总整个 stage 干预的样本级 CE、正确率、平均 G 和 G 分布。"""
    output = {}
    for name in METHODS:
        rows = [s["methods"][name] for s in samples]
        output[name] = {"mean_ce": float(np.mean([r["ce"] for r in rows])),
                        "accuracy": float(np.mean([r["correct"] for r in rows]))}
        if all("mean_g" in row for row in rows):
            output[name]["mean_g"] = float(np.mean([r["mean_g"] for r in rows]))
            output[name]["g_histogram"] = np.sum(
                [r["g_histogram"] for r in rows], axis=0).tolist()
    return output


def summarize_samples(samples: list[dict]) -> dict:
    """汇总样本级收益并校正多个预定义检验，完整保留原始 checkpoint 对照。"""
    tests = defaultdict(list)
    for sample in samples:
        probes = sample["probes"]
        rho = [p["spearman"] for p in probes if p["spearman"] is not None]
        if rho:
            tests["score_spearman"].append(float(np.mean(rho)))
        tests["top_bottom_gain"].append(float(np.mean([p["top_bottom_gain"] for p in probes])))
        methods = sample["methods"]
        for name in ("quality16", "quality4", "adaptive"):
            tests[f"{name}_vs_cosine16"].append(methods["cosine16"]["ce"] - methods[name]["ce"])
        for name in ("quality4", "adaptive"):
            tests[f"{name}_vs_quality16"].append(methods["quality16"]["ce"] - methods[name]["ce"])
    results = {key: sample_test(values) for key, values in tests.items()}
    ranked = sorted((r["p"], key) for key, r in results.items() if r["p"] is not None)
    previous = 1.0
    for rank in range(len(ranked), 0, -1):
        p, key = ranked[rank - 1]
        previous = min(previous, p * len(ranked) / rank)
        results[key]["q"] = previous
    accuracy = {name: float(np.mean([s["methods"][name]["correct"] for s in samples]))
                for name in METHODS}
    positive = lambda key: (key in results and results[key]["mean"] > 0 and
                            results[key]["q"] is not None and results[key]["q"] < .05)
    return {"tests": results, "stage_intervention_accuracy": accuracy,
            "method_metrics": method_metrics(samples),
            "original_checkpoint_mean_ce": float(np.mean([s["baseline"]["ce"] for s in samples])),
            "n_samples": len(samples),
            "min_possible_exact_two_sided_p": 2. ** (1 - len(samples)) if len(samples) <= 15 else None,
            "original_checkpoint_accuracy": float(np.mean([s["baseline"]["correct"] for s in samples])),
            "evidence": {"score_predicts_contribution": positive("score_spearman") and positive("top_bottom_gain"),
                         "quality16_improves_same_pool_cosine16": positive("quality16_vs_cosine16"),
                         "adaptive_improves_quality16": positive("adaptive_vs_quality16")},
            "note": "训练集内的特征发现留出样本；不是独立模型测试集准确率"}
