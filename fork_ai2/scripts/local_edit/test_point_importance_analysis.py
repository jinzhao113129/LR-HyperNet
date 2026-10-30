"""验证特征分析的配对、数据来源防错、样本聚类统计和输出。"""

import argparse
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import point_importance_analysis as analysis


def synthetic_capture() -> dict:
    """造出近处高贡献点、远处低贡献点及可区分的特征方向。"""
    xyz = np.zeros((40, 4), dtype=np.float32)
    xyz[:, 0] = np.linspace(0, 1, 40)
    xyz[20:, 1:3] = 2
    features = np.tile([1., 0., 0.], (40, 1)).astype(np.float32)
    features[20:] = [0., 2., 0.]
    return {"xyz": xyz, "features": features,
            "local": np.tile(np.arange(4), (2, 1)), "fps": np.array([0, 1])}


def synthetic_records() -> list[dict]:
    """生成八个独立样本，供样本层面配对统计。"""
    records = []
    for sample in range(8):
        pairs = [{"sample": sample, "center": center, "rank": rank,
                  "high_index": rank + 3, "low_index": rank + 23,
                  "high_gain": 1. / rank, "low_gain": -1.}
                 for center in range(2) for rank in range(1, 5)]
        records.extend(analysis.enrich_pairs(pairs, synthetic_capture()))
    return records


def test_clustered_statistics() -> None:
    """重复同一图像的点不应凭空增加独立样本量或显著性。"""
    records = synthetic_records()
    result = analysis.summarize(records)
    duplicate = analysis.summarize(records * 10)
    metric = result["neighbor_cosine"]
    assert metric["high"]["mean"] == 1
    assert metric["low"]["mean"] == 0
    assert metric["n_samples"] == 8
    assert metric["p"] == duplicate["neighbor_cosine"]["p"]
    assert metric["significant"]
    assert result["time_fraction"]["mean_difference"] < 0


def write_fixture(directory: Path) -> None:
    """保存选点记录和 NPZ，覆盖同轮低候选配对的实际文件格式。"""
    capture = synthetic_capture()
    ids = np.arange(4, 20)
    gains = 1 / np.arange(1, 17, dtype=np.float32)
    np.savez(directory / "sample0_center0.npz", indices=ids, gains=gains,
             xyz=capture["xyz"][ids].astype(np.float16),
             feature=capture["features"][ids].astype(np.float16))
    columns = ["sample", "center", "rank", "index", "gain"]
    for name, low in (("selected_points.csv", False),
                      ("low_contribution_points.csv", True)):
        with (directory / name).open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=columns)
            writer.writeheader()
            for rank in range(1, 17):
                writer.writerow({"sample": 0, "center": 0, "rank": rank,
                                 "index": rank + (23 if low else 3),
                                 "gain": -1 if low else float(gains[rank - 1])})
    (directory / "summary.json").write_text(
        json.dumps({"source": "train.h5 only"}), encoding="utf-8")


def test_sources_and_pairing() -> None:
    """阻止 train/test 错配，并核验 NPZ 与 CSV 以及模型特征对应。"""
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        write_fixture(directory)
        files = analysis.discover_files(directory)
        pairs = analysis.pairs_from_csv(directory, 4)
        assert len(pairs) == 4
        assert all(p["low_index"] == p["rank"] + 23 for p in pairs)
        analysis.verify_pairs(files, pairs)
        analysis.verify_artifact(files[0, 0], synthetic_capture())
        assert analysis.source_split(directory, None) == "train"
        try:
            analysis.source_split(directory, "test")
        except ValueError:
            pass
        else:
            raise AssertionError("必须拒绝 train/test 错配")
        changed = synthetic_capture()
        changed["features"] += 1
        try:
            analysis.verify_artifact(files[0, 0], changed)
        except ValueError:
            pass
        else:
            raise AssertionError("必须拒绝不同 checkpoint 的中间特征")


def test_coordinate_scaling() -> None:
    """负时间缩放也必须还原，不能把时间先后关系倒置。"""
    groups = [SimpleNamespace(fps_parameter=torch.tensor([[[2., -3., 4., 1.]]])),
              SimpleNamespace(fps_parameter=torch.tensor([[[-2., 1., .5, 1.]]]))]
    model = SimpleNamespace(local_grouper_list=groups)
    scale = analysis.coordinate_scale(model, 1)
    assert np.allclose(scale, [-4., -3., 2., 1.])
    raw = np.array([.8, .2, .3, 1.])
    assert np.allclose(raw * scale / scale, raw)


def test_output_without_gpu() -> None:
    """用合成重放结果跑完整输出流程，确认 JSON 字段与结论。"""
    original = analysis.replay_records
    analysis.replay_records = lambda *_args: (synthetic_records(), [])
    try:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_fixture(directory)
            output = directory / "result.json"
            args = argparse.Namespace(attribution_dir=str(directory),
                source_split=None, comparison="auto", top_k=4,
                inspect_only=False,
                log_dir="synthetic", use_ema=False, stage=1, output=str(output),
                ckpt="best", seed=42, data_path=str(directory))
            analysis.run(args)
            saved = json.loads(output.read_text(encoding="utf-8"))
            assert saved["comparison"] == "same_round"
            assert set(r["category"] for r in saved["metrics"].values()) == {
                "空间", "时间", "特征"}
            assert not list(directory.glob("*.png"))
    finally:
        analysis.replay_records = original


if __name__ == "__main__":
    test_clustered_statistics()
    test_sources_and_pairing()
    test_coordinate_scaling()
    test_output_without_gpu()
    print("point importance analysis PASS")
