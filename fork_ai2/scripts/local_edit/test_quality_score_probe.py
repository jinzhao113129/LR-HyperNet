"""验证综合分、数量边界、留出隔离、同池比较和真实聚合器干预。"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

# 本地仓 code/ 保存服务器 pi2 根目录模块；服务器本身无需这条补充路径。
local_code = Path(__file__).resolve().parents[3] / "code"
if local_code.is_dir():
    sys.path.append(str(local_code))

import quality_score_probe as probe
import quality_score_utils as core
from local_remote_grouper_v2 import LocalRemoteGrouper
from point_attribution_analysis import SubsetIntervention


class NumericTests(unittest.TestCase):
    """验证与模型无关的评分、停止规则和统计约束。"""

    def test_neighbor_cosine_is_average_pairwise(self):
        """模长不同的邻居不能先平均原始向量再计算余弦。"""
        features = np.array([[1., 0.], [0., 10.], [1., 0.]])
        values = core.quality_signals(features, np.zeros((3, 4)), 0,
                                      np.array([0, 1]), np.array([2]))
        self.assertAlmostEqual(values[0, 3], .5)
        self.assertAlmostEqual(values[0, 4], 1.)

    def test_constant_and_variable_counts(self):
        """常量信号选最小数，不同分数分布能够产生不同点数。"""
        score = core.quality_score(np.ones((32, 5)), np.ones(5))
        self.assertTrue(np.all(score == 0))
        self.assertEqual(len(core.adaptive_indices(score, 1, 16, .3, -1)), 1)
        four = core.adaptive_indices(np.array([4., 3., 2., 1., 0., -1.]), 1, 6, .3, -1)
        one = core.adaptive_indices(np.array([10., 0., -1., -2.]), 1, 4, .3, -1)
        self.assertEqual(len(four), 4)
        self.assertEqual(len(one), 1)
        self.assertEqual(len(core.adaptive_indices(-np.arange(1., 7.), 3, 6, .3, -1)), 3)
        with self.assertRaises(ValueError):
            core.adaptive_indices(score, 0, 16, .3, -1)

    def test_pool_and_discovery_isolation(self):
        """候选不重复且排除局部；发现样本与验证样本不得重叠。"""
        pool = core.sample_pool(80, np.arange(24), 32, [42, 1, 1, 0])
        self.assertEqual(len(set(pool)), 32)
        self.assertFalse(np.isin(pool, np.arange(24)).any())
        self.assertTrue(np.array_equal(pool, core.sample_pool(80, np.arange(24), 32, [42, 1, 1, 0])))
        ids = core.heldout_ids(20, [0, 1], None, 4)
        self.assertEqual(len(ids), 4)
        self.assertFalse(set(ids) & {0, 1})
        self.assertEqual(ids, core.heldout_ids(20, [0, 1], None, 4))
        with self.assertRaises(ValueError):
            core.heldout_ids(20, [0, 1], [1, 2], 2)

    def test_score_gain_statistics(self):
        """单调目标得到正相关，常量贡献不虚构相关系数。"""
        score = np.arange(20, dtype=float)
        stat = core.probe_statistics(score, score / 100)
        self.assertAlmostEqual(stat["spearman"], 1.)
        self.assertGreater(stat["top_bottom_gain"], 0)
        self.assertIsNone(core.probe_statistics(score, np.ones(20))["spearman"])

    def test_no_center_pseudoreplication(self):
        """复制同一图像的中心不能增加独立样本数或显著性。"""
        samples = []
        for index in range(8):
            methods = {name: {"ce": 2. if name == "cosine16" else 1., "correct": True}
                       for name in core.METHODS}
            samples.append({"probes": [{"spearman": .8, "top_bottom_gain": .1}],
                            "methods": methods, "baseline": {"correct": False, "ce": 3.}})
        first = core.summarize_samples(samples)
        for sample in samples:
            sample["probes"] *= 20
        second = core.summarize_samples(samples)
        for key in ("mean", "p", "q"):
            self.assertAlmostEqual(first["tests"]["score_spearman"][key],
                                   second["tests"]["score_spearman"][key])
        self.assertEqual(first["tests"]["score_spearman"]["n_samples"], 8)

    def test_discovery_report_provenance(self):
        """发现报告必须提供原始 train 行号，并与 checkpoint/stage/EMA 对齐。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "discovery.json"
            report = {"source_split": "train", "checkpoint": "cosine",
                      "stage": 0, "weight": "raw",
                      "artifact_checks": [{"source_sample_id": 3}, {"source_sample_id": 8}]}
            probe.write_json(path, report)
            args = argparse.Namespace(discovery_sample_ids=None, discovery_report=str(path),
                                      log_dir="cosine", stage=0, use_ema=False)
            self.assertEqual(probe.discovery_samples(args)[0], [3, 8])
            args.use_ema = True
            with self.assertRaises(ValueError):
                probe.discovery_samples(args)


class ToyModel(torch.nn.Module):
    """使用生产 LocalRemoteGrouper 的小模型，验证完整前向干预。"""

    def __init__(self):
        """使用8个中心、24个局部点和固定16个远程点。"""
        super().__init__()
        self.embed = torch.nn.Linear(4, 8)
        self.local_grouper_list = torch.nn.ModuleList([
            LocalRemoteGrouper(channel=8, groups=8, kneighbors=24,
                               remote_g=16, g16_treatment="cosine_near")])
        self.head = torch.nn.Linear(20, 3)

    def forward(self, inputs):
        """对真实分组输出做均值分类，保持输入为 B×4×N。"""
        coords = inputs.transpose(1, 2)
        _, values = self.local_grouper_list[0](coords, self.embed(coords))
        return self.head(values.mean(dim=(1, 2)))


class InterventionTests(unittest.TestCase):
    """用真实 Lorentz 聚合器检查基线回放、变长及探针缓存。"""

    def setUp(self):
        """只捕获 selector 输入输出，保留生产 forward 本身。"""
        torch.manual_seed(7)
        self.model = ToyModel().eval()
        self.group = self.model.local_grouper_list[0]
        self.capture = SimpleNamespace()
        self.handles = [self.group.remote_selector.register_forward_pre_hook(self.capture_input),
                        self.group.remote_selector.register_forward_hook(self.capture_output)]
        self.fps_capture = SubsetIntervention(self.group, self.capture)
        self.handles.append(self.group.register_forward_pre_hook(self.fps_capture.capture_input))
        self.intervention = probe.GroupIntervention(self.group, self.capture, self.fps_capture)
        self.handles.append(self.group.register_forward_hook(self.intervention.hook))
        self.x = torch.rand(1, 4, 64)
        self.baseline = probe.classify(self.model, self.x, 1, 42)
        self.original = self.capture.baseline_idx[0].cpu().numpy().copy()

    def capture_input(self, _module, inputs):
        """记录真实候选特征、局部点及中心特征。"""
        self.capture.center_feat, self.capture.all_feat, self.capture.local_idx = inputs[:3]

    def capture_output(self, _module, _inputs, output):
        """记录生产规则选择的原始16点。"""
        self.capture.baseline_idx = output

    def tearDown(self):
        """移除所有测试 hook。"""
        for handle in self.handles:
            handle.remove()

    def plans(self):
        """由实际特征和中心映射生成四规则共享的候选池。"""
        snapshot = probe.stage_snapshot(self.model, self.intervention, 0)
        args = argparse.Namespace(seed=42, stage=0, pool_size=32,
                                  weights=[1.] * 5, min_g=1, max_g=16,
                                  stop_ratio=.3, stop_abs=-1.)
        return [probe.center_plan(snapshot, center, 0, args) for center in range(8)]

    def test_baseline_and_four_rules(self):
        """原点集注回保持 logits 一致，四规则对完整 stage 有效。"""
        probe.verify_intervention(self.model, self.intervention, self.x, 1, 42,
                                   self.baseline, self.original)
        outcomes = probe.method_comparison(self.model, self.intervention, self.x,
                                           1, 42, self.plans())
        self.assertEqual(set(outcomes), set(core.METHODS))
        self.assertEqual(outcomes["quality16"]["mean_g"], 16)
        self.assertEqual(outcomes["quality4"]["mean_g"], 4)
        self.assertTrue(1 <= outcomes["adaptive"]["mean_g"] <= 16)
        self.assertTrue(all(np.isfinite(value["ce"]) for value in outcomes.values()))

    def test_probe_resume_and_fingerprint(self):
        """候选替换不改变16点基数，缓存可恢复且拒绝错误配置。"""
        plan = self.plans()[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "center.npz"
            first = probe.probe_center(self.model, self.intervention, self.x, 1, 42, 0,
                self.original[0], self.baseline, plan, path, "correct")
            self.assertGreaterEqual(first["n"], 4)
            with patch("quality_score_probe.classify", side_effect=AssertionError("缓存不应再前向")):
                repeated = probe.probe_center(self.model, self.intervention, self.x, 1, 42, 0,
                    self.original[0], self.baseline, plan, path, "correct")
            self.assertEqual(first, repeated)
            with self.assertRaises(ValueError):
                probe.probe_center(self.model, self.intervention, self.x, 1, 42, 0,
                    self.original[0], self.baseline, plan, path, "wrong")

    def test_complete_sample_flow(self):
        """用真实 FPS/聚合器完成两样本实验和整样本缓存恢复。"""
        args = argparse.Namespace(seed=42, stage=0, pool_size=32, weights=[1.] * 5,
            min_g=1, max_g=16, stop_ratio=.3, stop_abs=-1., centers_per_sample=1,
            cost_only=False, max_forwards=1000)
        data = {"data": np.repeat(self.x[0].T.numpy()[None], 4, axis=0),
                "label": np.array([[0], [0], [1], [2]])}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out = Path(folder)
            manifest = {"test": "fixed_real_grouper", "sample_ids": [2, 3]}
            probe.run_loaded(args, self.model, self.intervention, data, [2, 3],
                              manifest, torch.device("cpu"), out)
            result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(result["stage_intervention_accuracy"]), set(core.METHODS))
            self.assertEqual(result["tests"]["top_bottom_gain"]["n_samples"], 2)
            with patch("quality_score_probe.evaluate_sample", side_effect=AssertionError("应恢复缓存")):
                probe.run_loaded(args, self.model, self.intervention, data, [2, 3],
                                  manifest, torch.device("cpu"), out)

    def test_cost_pilot(self):
        """成本探针只跑三个候选，默认 CPU 不查询训练显卡。"""
        args = argparse.Namespace(seed=42, stage=0, pool_size=32, weights=[1.] * 5,
                                  min_g=1, max_g=16, stop_ratio=.3, stop_abs=-1.)
        pilot = probe.cost_pilot(self.model, self.intervention, self.x[0].T.numpy(),
                                  1, 0, args, torch.device("cpu"), 100)
        self.assertEqual(pilot["pilot_forwards"], 3)
        self.assertGreater(pilot["estimated_probe_hours"], 0)
        self.assertIsNone(pilot["cuda_peak_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
