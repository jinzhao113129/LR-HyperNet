"""验证 adaptive_g 的 Torch/Numpy 一致性、真实长度聚合和两轮合成训练。"""

import argparse
import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from local_remote_grouper_v2 import LocalRemoteGrouper, RemoteSelector
from quality_score_utils import adaptive_indices, quality_score, quality_signals


class AdaptiveSelectorTests(unittest.TestCase):
    """验证五信号、去重、阈值及跨中心分块行为。"""

    def test_signals_against_numpy(self):
        """相同池/坐标下 Torch 五信号和质量分应与参考实现一致。"""
        torch.manual_seed(9)
        selector = RemoteSelector(g16_treatment='adaptive_g', channel=8)
        feature = torch.randn(1, 64, 8)
        xyz = torch.rand(1, 64, 3)
        local = torch.arange(24).reshape(1, 1, 24)
        pool = torch.arange(24, 56).reshape(1, 1, 32)
        signals = selector._adaptive_signals(feature[:, :1], feature, local,
                                               xyz, xyz[:, :1], pool)
        expected = quality_signals(feature[0].numpy(), xyz[0].numpy(), 0,
                                   local[0, 0].numpy(), pool[0, 0].numpy())
        np.testing.assert_allclose(signals[0, 0].numpy(), expected, rtol=1e-5, atol=1e-6)
        valid = torch.ones_like(pool, dtype=torch.bool)
        actual = selector._adaptive_quality(signals, valid)
        np.testing.assert_allclose(actual[0, 0].numpy(), quality_score(expected, np.ones(5)),
                                   rtol=1e-5, atol=2e-6)
        duplicated = torch.cat([signals, signals[..., :8, :]], dim=-2)
        mask = torch.cat([valid, torch.zeros_like(valid[..., :8])], dim=-1)
        repeated = selector._adaptive_quality(duplicated, mask)
        torch.testing.assert_close(repeated[..., :32], actual)

    def test_different_counts_and_padding(self):
        """两个中心分别选4点和1点，填充位复制末点，不算有效点。"""
        selector = RemoteSelector(g16_treatment='adaptive_g', adaptive_max_g=6)
        values = torch.tensor([[[4., 3., 2., 1., 0., -1.],
                                [10., 0., -1., -2., -3., -4.]]])
        pool = torch.arange(6).repeat(1, 2, 1)
        indices, counts = selector._adaptive_rank(pool, values, torch.ones_like(pool).bool())
        self.assertEqual(counts.tolist(), [[4, 1]])
        self.assertEqual(indices[0, 0].tolist(), [0, 1, 2, 3, 3, 3])
        self.assertEqual(indices[0, 1].tolist(), [0] * 6)
        for center in range(2):
            expected = adaptive_indices(values[0, center].numpy(), 1, 6, .3, -1.)
            np.testing.assert_array_equal(indices[0, center, :counts[0, center]].numpy(), expected)
        _, flat = selector._adaptive_rank(pool, torch.zeros_like(values), torch.ones_like(pool).bool())
        self.assertEqual(flat.tolist(), [[1, 1]])

    def test_chunking_and_local_exclusion(self):
        """超过64个中心时仍不重复、不选局部点，并缓存真实长度。"""
        torch.manual_seed(11)
        selector = RemoteSelector(g16_treatment='adaptive_g', channel=8)
        feature, xyz = torch.randn(2, 96, 8), torch.rand(2, 96, 3)
        local = torch.arange(24).repeat(2, 70, 1)
        selector._g16_sink = []
        output = selector(feature[:, :70], feature, local,
                           all_xyz=xyz, center_xyz=xyz[:, :70])
        counts = selector.last_selected_counts
        self.assertEqual(tuple(output.shape), (2, 70, 16))
        self.assertTrue(((counts >= 1) & (counts <= 16)).all())
        for batch in range(2):
            for center in range(70):
                valid = output[batch, center, :counts[batch, center]]
                self.assertEqual(valid.unique().numel(), valid.numel())
                self.assertFalse(torch.isin(valid, local[batch, center]).any())
        self.assertTrue(torch.equal(selector._g16_sink[0]['valid_counts'], counts))
        stricter = RemoteSelector(g16_treatment='adaptive_g', channel=8,
                                   adaptive_stop_ratio=.9)
        stricter(feature[:, :70], feature, local,
                 all_xyz=xyz, center_xyz=xyz[:, :70])
        self.assertGreater(stricter.last_selected_counts.unique().numel(), 1)

    def test_invalid_configuration(self):
        """非法数量、阈值和权重在构造时直接报错。"""
        configurations = [{'adaptive_min_g': 0}, {'adaptive_max_g': 17},
            {'adaptive_stop_ratio': float('nan')}, {'adaptive_stop_ratio': -1},
            {'adaptive_stop_abs': float('inf')}, {'adaptive_weights': (1, 2)},
            {'adaptive_weights': (1, 1, -1, 1, 1)}]
        for config in configurations:
            with self.assertRaises(ValueError):
                RemoteSelector(g16_treatment='adaptive_g', **config)


class AdaptiveAggregationTests(unittest.TestCase):
    """用生产聚合器验证重复填充不会改变结果或梯度。"""

    def test_padding_invariance_and_backward(self):
        """真实点数不同的中心按各自点集聚合；填充张量的梯度恒零。"""
        torch.manual_seed(17)
        group = LocalRemoteGrouper(channel=8, groups=2, g16_treatment='adaptive_g',
                                   remote_stochastic_depth=0., hyperbolic_type='lorentz').eval()
        group.remote_selector.last_selected_counts = torch.tensor([[1, 4]])
        points = torch.randn(1, 2, 16, 8, requires_grad=True)
        result = group._aggregate_adaptive(points)
        expected = torch.stack([group.hyper_agg(points[:, 0:1, :1])[:, 0],
                                group.hyper_agg(points[:, 1:2, :4])[:, 0]], dim=1)
        torch.testing.assert_close(result, expected)
        changed = points.detach().clone()
        changed[:, 0, 1:] = 1000
        changed[:, 1, 4:] = -1000
        torch.testing.assert_close(group._aggregate_adaptive(changed), result)
        result.square().mean().backward()
        self.assertEqual(points.grad[:, 0, 1:].abs().sum().item(), 0)
        self.assertEqual(points.grad[:, 1, 4:].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertIsNotNone(group.hyper_agg.out_proj.weight.grad)

    def test_local_grouper_train_path(self):
        """完整分组器含 stochastic depth 的训练路径可以正常反传。"""
        torch.manual_seed(23)
        group = LocalRemoteGrouper(channel=8, groups=8, g16_treatment='adaptive_g',
                                   hyperbolic_type='lorentz').train()
        features = torch.randn(2, 64, 8, requires_grad=True)
        _, output = group(torch.rand(2, 64, 4), features)
        self.assertEqual(tuple(output.shape), (2, 8, 24, 20))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)


class AdaptiveModelTests(unittest.TestCase):
    """验证 argparse→FECNet→grouper→selector 和 checkpoint 重建。"""

    def test_training_arguments(self):
        """直接执行生产 argparse 函数，避免导入训练模块启动环境副作用。"""
        path = Path(__file__).with_name('train_frequency_ncaltech.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'parse_args')
        namespace = {'argparse': argparse}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        with patch.object(sys, 'argv', ['train', '--remote_g16', '--g16_treatment', 'adaptive_g',
             '--adaptive_min_g', '3', '--adaptive_max_g', '7', '--adaptive_weights', '1', '0', '2', '0', '1']):
            args = namespace['parse_args']()
        self.assertEqual(args.adaptive_min_g, 3)
        self.assertEqual(args.adaptive_max_g, 7)
        self.assertEqual(args.adaptive_weights, [1., 0., 2., 0., 1.])
        from FECNet import FECNet
        with self.assertRaises(ValueError):
            FECNet(num_classes=3, g16_treatment='adaptive_g')

    def test_two_epoch_synthetic_training(self):
        """真实 FECNet 三个 stage 连续训练两轮，每轮两个 batch。"""
        from FECNet import FECNet
        torch.manual_seed(29)
        model = FECNet(num_classes=3, feature_list=[8, 20, 44, 92],
            group_number=[128, 64, 32, 16], use_remote_g16=True,
            g16_treatment='adaptive_g', adaptive_min_g=2, adaptive_max_g=7,
            adaptive_stop_ratio=.4, adaptive_weights=(1., 0., 2., 0., 1.)).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for epoch in range(2):
            for batch in range(2):
                optimizer.zero_grad()
                output = model(torch.rand(2, 4, 256))
                loss = F.cross_entropy(output, torch.tensor([0, 2]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
        for group in model.local_grouper_list:
            selector = group.remote_selector
            self.assertEqual(selector.adaptive_weights, (1., 0., 2., 0., 1.))
            self.assertTrue(((selector.last_selected_counts >= 2) &
                             (selector.last_selected_counts <= 7)).all())
        clone = FECNet(num_classes=3, feature_list=[8, 20, 44, 92],
            group_number=[128, 64, 32, 16], use_remote_g16=True, g16_treatment='adaptive_g')
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_checkpoint_reconstruction(self):
        """生产重建函数恢复非默认参数，旧 checkpoint 缺少新键时仍可严格加载。"""
        from FECNet import FECNet
        path = Path(__file__).with_name('extract_preds.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        flags = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'FLAG_KEYS' for t in n.targets))
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_model')
        namespace = {'FECNet': FECNet}
        exec(compile(ast.Module(body=[flags, build], type_ignores=[]), str(path), 'exec'), namespace)
        saved = {key: False for key in namespace['FLAG_KEYS']}
        saved.update(remote_g16=True, g16_treatment='adaptive_g', base_dim=8, fps_groups=128,
                     adaptive_min_g=3, adaptive_max_g=6, adaptive_stop_ratio=.7,
                     adaptive_stop_abs=-.2, adaptive_weights=[1, 0, 0, 2, 1])
        model, _ = namespace['build_model'](saved.copy(), 3)
        for group in model.local_grouper_list:
            selector = group.remote_selector
            self.assertEqual(selector.adaptive_min_g, 3)
            self.assertEqual(selector.adaptive_max_g, 6)
            self.assertEqual(selector.adaptive_stop_ratio, .7)
            self.assertEqual(selector.adaptive_stop_abs, -.2)
            self.assertEqual(selector.adaptive_weights, (1., 0., 0., 2., 1.))
        old = {key: value for key, value in saved.items() if not key.startswith('adaptive_')}
        old['g16_treatment'] = 'cosine_near'
        legacy, _ = namespace['build_model'](old, 3)
        legacy.load_state_dict(model.state_dict(), strict=True)

    def test_batch_count_logging(self):
        """日志按有效中心加权，不把重复填充位当成16点。"""
        path = Path(__file__).with_name('train_frequency_ncaltech.py')
        function = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'print_adaptive_counts')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        from types import SimpleNamespace
        selector = SimpleNamespace(g16_treatment='adaptive_g',
                                    last_selected_counts=torch.tensor([[1, 4]]))
        model = SimpleNamespace(local_grouper_list=[SimpleNamespace(remote_selector=selector)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            namespace['print_adaptive_counts'](model, 7)
        self.assertIn('mean_selected_g=2.5000', output.getvalue())


if __name__ == '__main__':
    unittest.main()
