"""验证全点扫描缓存、候选池定义及小池精确搜索的正确性。"""

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

# 本地仓只保存 GPU 部署文件的平铺副本，导入依赖在服务器的 pi2 布局中。
attribution = types.ModuleType("point_attribution_analysis")
attribution.SubsetIntervention = object
attribution.loss_and_logit = lambda *_args: (0.0, 0.0)
probe = types.ModuleType("selector_utility_probe")
probe.SelectorIntervention = object
probe.load_model = lambda *_args: None
probe.predict = lambda *_args: None
probe.train_examples = lambda *_args: None
sys.modules["point_attribution_analysis"] = attribution
sys.modules["selector_utility_probe"] = probe

import point_subset_search as search


def test_exact_pool() -> None:
    """构造 18 点可解析目标，验证穷举结果与续跑缓存。"""
    losses = np.arange(18, dtype=np.float32)
    baseline = np.arange(2, 18)
    pool = search.make_pool(baseline, losses, 18)
    assert pool == list(range(18))
    original = search.evaluate
    search.evaluate = lambda _m, _i, _x, _y, _s, subset: (
        float(sum(subset)), 0.0)
    try:
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "progress.json"
            best, value, _ = search.exact_search(
                None, None, None, 0, 0, pool, state, {"id": 1}, 20)
            assert best == list(range(16))
            assert value == sum(range(16))
            again, repeated, _ = search.exact_search(
                None, None, None, 0, 0, pool, state, {"id": 1}, 20)
            assert again == best and repeated == value
            saved = json.loads(state.read_text(encoding="utf-8"))
            assert saved["next"] == 153
    finally:
        search.evaluate = original


def test_scan_resume() -> None:
    """验证全候选扫描会存档，且缓存与元数据不一致时报错。"""
    original = search.evaluate
    search.evaluate = lambda _m, _i, _x, _y, _s, subset: (
        float(subset[0] ** 2), 0.0)
    try:
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "scan.npz"
            first, _ = search.scan_all(None, None, None, 0, 0, 18, cache,
                                       {"id": 1}, 5)
            second, _ = search.scan_all(None, None, None, 0, 0, 18, cache,
                                        {"id": 1}, 5)
            assert np.array_equal(first, second)
            assert first[17] == 289
            try:
                search.scan_all(None, None, None, 0, 0, 18, cache,
                                {"id": 2}, 5)
            except ValueError:
                pass
            else:
                raise AssertionError("应拒绝不同 checkpoint 的缓存")
    finally:
        search.evaluate = original


if __name__ == "__main__":
    test_exact_pool()
    test_scan_resume()
    print("point subset exact search PASS")
