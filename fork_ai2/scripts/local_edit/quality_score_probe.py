"""在特征发现之外的训练样本上验证综合分，比较四种同池选点规则。

不训练。相关性探针每次仅换一个中心的第16点，固定另外15点；
四规则则同时替换一个 stage 的全部中心，其余 stage 由完整前向执行。
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from quality_score_utils import (METHODS, SIGNALS, heldout_ids, probe_statistics,
                                 quality_score, quality_signals, rule_sets,
                                 sample_pool, summarize_samples)


class GroupIntervention:
    """仅替换指定 stage 的 remote 通道，按真实 G 分组复用原聚合器。"""

    def __init__(self, grouper, selector, capture):
        """复用已有 FPS 重放器和 selector 捕获接口。"""
        self.grouper = grouper
        self.selector = selector
        self.capture = capture
        self.replacements: dict[int, list[int]] = {}

    def hook(self, _module, _inputs, output):
        """按实际长度聚合，避免填充污染注意力、均值和残差。"""
        if not self.replacements:
            return output
        xyz, grouped = output
        width = self.selector.all_feat.shape[-1]
        changed = grouped.clone()
        lengths = {len(ids) for ids in self.replacements.values()}
        for length in sorted(lengths):
            if not 1 <= length <= 16:
                raise ValueError("干预点数须在 [1,16] 内")
            centers = sorted(c for c, ids in self.replacements.items() if len(ids) == length)
            ids = [self.replacements[c] for c in centers]
            if any(len(set(row)) != len(row) for row in ids):
                raise ValueError("同一中心不得重复选点")
            index = torch.as_tensor(ids, device=grouped.device)
            points = self.selector.all_feat[:, index]
            remote = self.grouper.hyper_agg(points)
            changed[:, centers, :, -width:] = remote.unsqueeze(-2)
        return xyz, changed


def classify(model, x, label: int, seed: int) -> dict:
    """用同一全局 FPS 随机种子完整前向，返回 CE、margin 和预测。"""
    torch.manual_seed(seed)
    if x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        logits = model(x)
        target = torch.tensor([label], device=x.device)
        ce = float(F.cross_entropy(logits, target))
        others = logits[0].clone()
        others[label] = -torch.inf
        result = {"ce": ce, "margin": float(logits[0, label] - others.max()),
                  "target_logit": float(logits[0, label]),
                  "prediction": int(logits.argmax(-1)[0]),
                  "correct": bool(logits.argmax(-1)[0] == label),
                  "logits": logits[0].cpu().tolist()}
    if not np.isfinite(ce):
        raise RuntimeError("前向 CE 非有限")
    return result


def stage_snapshot(model, intervention, stage: int) -> dict:
    """缓存 stage 特征并还原累计坐标缩放，以对齐五项特征分析。"""
    selector, capture = intervention.selector, intervention.capture
    convert = lambda value: value.detach().cpu().numpy().copy()
    scale = np.ones(4)
    for group in model.local_grouper_list[:stage + 1]:
        scale *= convert(group.fps_parameter).reshape(4)
    if np.any(np.abs(scale) < 1e-8):
        raise ValueError("坐标累计缩放接近零，无法还原单位")
    snapshot = {"features": convert(selector.all_feat[0]),
                "xyz": convert(capture.input_xyz[0]) / scale,
                "local": convert(selector.local_idx[0]),
                "fps": convert(capture.fps_idx[0]),
                "baseline": convert(selector.baseline_idx[0])}
    centers = snapshot["features"][snapshot["fps"]]
    if not np.allclose(centers, convert(selector.center_feat[0]), rtol=1e-5, atol=1e-5):
        raise ValueError("FPS 重放与实际 selector 中心不一致")
    if snapshot["baseline"].shape[1] != 16:
        raise ValueError("本探针要求原 checkpoint 每中心固定16点")
    return snapshot


def center_plan(snapshot: dict, center: int, sample: int, args) -> dict:
    """四规则共享均匀无放回候选池，不按标签或固定距离预筛。"""
    pool = sample_pool(len(snapshot["features"]), snapshot["local"][center],
                       args.pool_size, [args.seed, sample, args.stage, center])
    signals = quality_signals(snapshot["features"], snapshot["xyz"],
                              int(snapshot["fps"][center]),
                              snapshot["local"][center], pool)
    scores = quality_score(signals, np.asarray(args.weights))
    sets = rule_sets(pool, signals, scores, args.min_g, args.max_g,
                     args.stop_ratio, args.stop_abs)
    return {"pool": pool, "signals": signals, "scores": scores, "sets": sets}


def save_probe(path: Path, plan: dict, gains: np.ndarray, margins: np.ndarray,
               context: list[int], fingerprint: str) -> None:
    """定期保存单中心替换进度和五项分数，不使用 pickle。"""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, pool=plan["pool"], signals=plan["signals"],
                            scores=plan["scores"], gains=gains, margin_gains=margins,
                            context=np.asarray(context), fingerprint=np.array(fingerprint))
    temporary.replace(path)


def probe_center(model, intervention, x, label, seed, center,
                  reference, baseline, plan, path, fingerprint) -> dict:
    """固定基线15点，逐个换入候选；已在上下文内的点不重复加入。"""
    context = sorted(int(i) for i in reference[:-1])
    gains = np.full(len(plan["pool"]), np.nan)
    margins = gains.copy()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["fingerprint"]) != fingerprint or
                not np.array_equal(saved["pool"], plan["pool"]) or
                not np.array_equal(saved["context"], context) or
                not np.allclose(saved["scores"], plan["scores"])):
                raise ValueError("中心缓存与当前实验配置不匹配")
            gains, margins = saved["gains"].copy(), saved["margin_gains"].copy()
    eligible = ~np.isin(plan["pool"], context)
    for position in np.flatnonzero(eligible):
        if np.isfinite(gains[position]):
            continue
        ids = sorted(context + [int(plan["pool"][position])])
        intervention.replacements = {center: ids}
        result = classify(model, x, label, seed)
        gains[position] = baseline["ce"] - result["ce"]
        margins[position] = result["margin"] - baseline["margin"]
        if position % 8 == 0:
            save_probe(path, plan, gains, margins, context, fingerprint)
    save_probe(path, plan, gains, margins, context, fingerprint)
    intervention.replacements = {}
    stats = probe_statistics(plan["scores"], gains)
    return {"center": center, "context_15": context, **stats}


def verify_intervention(model, intervention, x, label, seed,
                        baseline, original: np.ndarray) -> None:
    """把全部基线16点原样注回，硬校验干预不改变模型输出。"""
    intervention.replacements = {center: ids.tolist() for center, ids in enumerate(original)}
    repeated = classify(model, x, label, seed)
    intervention.replacements = {}
    if not np.allclose(baseline["logits"], repeated["logits"], rtol=1e-5, atol=1e-5):
        raise RuntimeError("基线16点注回校验失败，不执行贡献分析")


def method_comparison(model, intervention, x, label, seed, plans) -> dict:
    """对整个指定 stage 同时使用四规则，报告每张样本的完整预测。"""
    outcomes = {}
    for name in METHODS:
        intervention.replacements = {center: plan["sets"][name]
                                     for center, plan in enumerate(plans)}
        outcomes[name] = classify(model, x, label, seed)
        counts = [len(plan["sets"][name]) for plan in plans]
        outcomes[name]["mean_g"] = float(np.mean(counts))
        outcomes[name]["g_histogram"] = np.bincount(counts, minlength=17).tolist()
    intervention.replacements = {}
    return outcomes


def evaluate_sample(model, intervention, raw, label, sample, args,
                     device, out, fingerprint) -> dict:
    """一次原始前向后构建同池规则、测单点贡献及整 stage 四规则。"""
    seed = args.seed + sample
    x = torch.from_numpy(np.asarray(raw).reshape(-1, 4)).float().T.unsqueeze(0).to(device)
    intervention.replacements = {}
    baseline = classify(model, x, int(label), seed)
    snapshot = stage_snapshot(model, intervention, args.stage)
    verify_intervention(model, intervention, x, int(label), seed,
                        baseline, snapshot["baseline"])
    plans = [center_plan(snapshot, center, sample, args)
             for center in range(len(snapshot["fps"]))]
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, sample, 771]))
    count = min(args.centers_per_sample, len(plans))
    centers = np.sort(rng.choice(len(plans), count, replace=False))
    probes = []
    for center in centers:
        center = int(center)
        path = out / f"sample{sample}_center{center}.npz"
        probes.append(probe_center(model, intervention, x, int(label), seed, center,
            snapshot["baseline"][center], baseline, plans[center], path, fingerprint))
        print(f"sample={sample} center={center} {probes[-1]}", flush=True)
    return {"sample_id": sample, "label": int(label), "baseline": baseline,
            "probes": probes, "stage_centers": len(plans),
            "methods": method_comparison(model, intervention, x, int(label), seed, plans)}


def discovery_samples(args) -> tuple[list[int], str]:
    """读取原始 H5 发现样本编号，不猜测 NPZ 中重编号的 sample 含义。"""
    if args.discovery_sample_ids is not None:
        return args.discovery_sample_ids, "explicit_original_train_row_ids"
    path = Path(args.discovery_report).expanduser()
    if not path.is_file():
        raise ValueError("缺少特征发现报告；请用 --discovery_sample_ids 指定原 train 行号")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("source_split") != "train":
        raise ValueError("发现报告必须声明 source_split=train")
    expected = {"checkpoint": args.log_dir, "stage": args.stage,
                "weight": "ema" if args.use_ema else "raw"}
    for key, value in expected.items():
        if key in report and report[key] != value:
            raise ValueError(f"发现报告的 {key} 与当前验证不一致")
    checks = report.get("artifact_checks", [])
    if not checks or any("source_sample_id" not in row for row in checks):
        raise ValueError("报告缺少原始样本行号；请显式指定 --discovery_sample_ids")
    return sorted({int(row["source_sample_id"]) for row in checks}), str(path)


def file_hash(path: Path) -> str:
    """计算小型代码/checkpoint 文件指纹，避免重启时复用错误缓存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """原子替换本工具生成的 JSON，避免中断损坏已有缓存。"""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def loaded_code_hashes(model, stage: int) -> dict:
    """记录实际加载的模型源文件，防止 tools/models 遮蔽导致缓存误用。"""
    from modules import furthest_point_sample
    group = model.local_grouper_list[stage]
    objects = (type(model), type(group), type(group.remote_selector),
               type(group.hyper_agg), furthest_point_sample)
    paths = {Path(inspect.getfile(obj)).resolve() for obj in objects}
    return {str(path): file_hash(path) for path in sorted(paths)}


def manifest_for(args, ids, discovery, source, data_path, checkpoint, model) -> dict:
    """记录样本划分、权重、阈值与代码版本，恢复时必须完全相符。"""
    config = {key: getattr(args, key) for key in
              ("log_dir", "ckpt", "use_ema", "num_category", "stage", "seed", "pool_size",
               "centers_per_sample", "weights", "min_g", "max_g", "stop_ratio", "stop_abs")}
    stat = data_path.stat()
    root = Path(__file__).resolve().parent
    return {"config": config, "sample_ids": ids, "discovery_sample_ids": discovery,
            "discovery_source": source, "split": "train",
            "checkpoint_sha256": file_hash(checkpoint),
            "dataset": {"path": str(data_path.resolve()), "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns},
            "code": {name: file_hash(root / name) for name in
                     ("quality_score_probe.py", "quality_score_utils.py",
                      "point_attribution_analysis.py", "selector_utility_probe.py")},
            "loaded_model_code": loaded_code_hashes(model, args.stage),
            "pool": "uniform_without_replacement_excluding_local_only",
            "scope": "one stage, all FPS centers for method accuracy"}


def prepare_output(out: Path, manifest: dict) -> str:
    """创建输出目录；不同实验配置不可混在一个目录恢复。"""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("输出目录已有不同配置的结果，请使用新的 --out_dir")
    write_json(path, manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def cost_pilot(model, intervention, raw, label, sample, args, device, planned):
    """只测少量替换前向，估算贡献探针成本，不启动完整扫描。"""
    x = torch.from_numpy(np.asarray(raw).reshape(-1, 4)).float().T.unsqueeze(0).to(device)
    seed = args.seed + sample
    classify(model, x, int(label), seed)
    snapshot = stage_snapshot(model, intervention, args.stage)
    plan = center_plan(snapshot, 0, sample, args)
    context = snapshot["baseline"][0, :15].tolist()
    candidates = [int(i) for i in plan["pool"] if i not in context][:3]
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index in candidates:
        intervention.replacements = {0: sorted(context + [index])}
        classify(model, x, int(label), seed)
    if x.is_cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    intervention.replacements = {}
    return {"pilot_forwards": len(candidates), "seconds": seconds,
            "estimated_probe_hours": seconds / len(candidates) * planned / 3600,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if x.is_cuda else None,
            "note": "仅估计替换前向；整stage信号构建/变长聚合额外成本尚未计入"}


def run_loaded(args, model, intervention, data, ids, manifest, device, out):
    """按样本存档恢复，最终汇总留出样本上的分数和四规则结果。"""
    planned = len(ids) * (args.centers_per_sample * args.pool_size + 6)
    if args.cost_only:
        result = cost_pilot(model, intervention, data["data"][ids[0]],
            int(np.asarray(data["label"][ids[0]]).item()), ids[0], args, device, planned)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "pilot.json", {"manifest": manifest, **result})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if planned > args.max_forwards:
        raise ValueError(f"预计约 {planned} 次完整前向，超过 --max_forwards；先 --cost_only")
    fingerprint = prepare_output(out, manifest)
    samples = []
    for sample in ids:
        path = out / f"sample{sample}_result.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["fingerprint"] != fingerprint:
                raise ValueError("样本缓存指纹不匹配")
            result = cached["result"]
        else:
            label = int(np.asarray(data["label"][sample]).item())
            result = evaluate_sample(model, intervention, data["data"][sample], label,
                                      sample, args, device, out, fingerprint)
            write_json(path, {"fingerprint": fingerprint, "result": result})
        samples.append(result)
    summary = {"manifest": manifest, **summarize_samples(samples),
               "limits": ["固定模型的前向干预不能替代完整训练", "一个基线15点上下文不代表所有组合",
                          "cosine16来自同一随机候选池，不等于原checkpoint的cosine全局选择"]}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run(args) -> None:
    """只打开 train.h5，固定模型并安装可撤销的 stage 干预。"""
    from point_attribution_analysis import SubsetIntervention
    from selector_utility_probe import ROOT, SelectorIntervention, load_model
    path = Path(args.data_path).expanduser() / "train.h5"
    discovery, source = discovery_samples(args)
    checkpoint = ROOT / "log" / "classification" / args.log_dir / "checkpoints" / f"{args.ckpt}_checkpoint.pth"
    device = torch.device(args.device)
    model = load_model(args, device)
    grouper = model.local_grouper_list[args.stage]
    if grouper.remote_selector.g16_treatment in {"random", "spatial_fixed"}:
        raise ValueError("请选择确定性固定 G16 checkpoint")
    if grouper.use_remote_gate or grouper.use_global_anchor or grouper.remote_kda_enc:
        raise ValueError("当前干预不支持 remote gate/anchor/KDA")
    selector = SelectorIntervention(grouper.remote_selector)
    selector.install()
    capture = SubsetIntervention(grouper, selector)
    intervention = GroupIntervention(grouper, selector, capture)
    pre = grouper.register_forward_pre_hook(capture.capture_input)
    post = grouper.register_forward_hook(intervention.hook)
    try:
        with h5py.File(path, "r") as data:
            ids = heldout_ids(len(data["data"]), discovery, args.sample_ids,
                              args.max_samples, args.seed)
            manifest = manifest_for(args, ids, discovery, source, path, checkpoint, model)
            run_loaded(args, model, intervention, data, ids, manifest, device,
                       Path(args.out_dir).expanduser())
    finally:
        pre.remove()
        post.remove()
        selector.remove()


def parse_args():
    """默认四样本两中心仅供小规模筛查；大规模前先用 cost_only 测成本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", default="secnet_pi2_g16_cosine_baidu")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--num_category", type=int, default=101)
    parser.add_argument("--data_path", default="./data/")
    parser.add_argument("--discovery_report", default="~/zhaojin/logs_run/point_importance_analysis.json")
    parser.add_argument("--discovery_sample_ids", nargs="+", type=int)
    parser.add_argument("--sample_ids", nargs="+", type=int)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--centers_per_sample", type=int, default=2)
    parser.add_argument("--pool_size", type=int, default=128)
    parser.add_argument("--weights", nargs=5, type=float, default=[1.] * 5)
    parser.add_argument("--stop_ratio", type=float, default=.3)
    parser.add_argument("--stop_abs", type=float, default=-1.)
    parser.add_argument("--min_g", type=int, default=1)
    parser.add_argument("--max_g", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0；默认不占训练显卡")
    parser.add_argument("--max_forwards", type=int, default=5000)
    parser.add_argument("--cost_only", action="store_true")
    parser.add_argument("--out_dir", default="~/zhaojin/logs_run/quality_score_validation")
    args = parser.parse_args()
    if args.centers_per_sample < 1 or args.max_samples < 1 or args.pool_size < 20:
        parser.error("centers_per_sample/max_samples 须为正，pool_size 须至少20")
    rule_sets(np.arange(args.pool_size), np.zeros((args.pool_size, 5)),
              quality_score(np.zeros((args.pool_size, 5)), np.array(args.weights)),
              args.min_g, args.max_g, args.stop_ratio, args.stop_abs)
    return args


if __name__ == "__main__":
    run(parse_args())
