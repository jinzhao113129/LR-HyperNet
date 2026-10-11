"""双曲原型分类头 CPU 验证脚本。

验证：
1. HyperbolicPrototypeClassifier 前向输出 shape 正确
2. 反向无 NaN/Inf
3. LRHyperNet(hyperbolic_classifier=True) 端到端前向+反向
4. 参数量统计
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from models.local_remote_grouper import HyperbolicPrototypeClassifier, lorentz_exp_origin, lorentz_distance


def test_classifier_forward():
    """测试分类头前向"""
    print("=== Test 1: HyperbolicPrototypeClassifier forward ===")
    clf = HyperbolicPrototypeClassifier(input_dim=1080, num_classes=101, embed_dim=64)
    x = torch.randn(4, 1080)
    out = clf(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == (4, 101), f"Expected (4, 101), got {out.shape}"
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    assert torch.isfinite(out).all(), "Output has NaN/Inf"
    print("  PASS")


def test_classifier_backward():
    """测试分类头反向"""
    print("\n=== Test 2: HyperbolicPrototypeClassifier backward ===")
    clf = HyperbolicPrototypeClassifier(input_dim=1080, num_classes=101, embed_dim=64)
    x = torch.randn(4, 1080, requires_grad=True)
    out = clf(x)
    loss = out.sum()
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  x.grad finite: {torch.isfinite(x.grad).all().item()}")
    print(f"  prototypes.grad finite: {torch.isfinite(clf.prototypes.grad).all().item()}")
    print(f"  curvature.grad: {clf.curvature.grad.item():.6f}")
    assert torch.isfinite(x.grad).all(), "x.grad has NaN/Inf"
    assert torch.isfinite(clf.prototypes.grad).all(), "prototypes.grad has NaN/Inf"
    print("  PASS")


def test_lorentz_functions():
    """测试 Lorentz 函数"""
    print("\n=== Test 3: Lorentz functions ===")
    v = torch.randn(8, 64)
    c = torch.tensor(0.4)
    x = lorentz_exp_origin(v, c)
    print(f"  exp_origin: input {v.shape} -> output {x.shape}")
    # 验证流形约束: x0^2 - ||x[1:]||^2 = 1/c
    constraint = x[..., 0]**2 - (x[..., 1:]**2).sum(-1)
    print(f"  Manifold constraint x0^2-||x||^2 = 1/c: {constraint.mean().item():.6f} (expected {1/0.4:.6f})")
    # 测试距离
    y = lorentz_exp_origin(torch.randn(8, 64), c)
    dist = lorentz_distance(x, y, c)
    print(f"  Distance range: [{dist.min().item():.4f}, {dist.max().item():.4f}]")
    assert torch.isfinite(dist).all(), "Distance has NaN/Inf"
    print("  PASS")


def test_full_model():
    """测试完整 LRHyperNet + 双曲分类头"""
    print("\n=== Test 4: Full LRHyperNet(hyperbolic_classifier=True) ===")
    from lr_hypernet import LRHyperNet
    model = LRHyperNet(
        num_classes=101,
        hyperbolic_type="euclidean",  # 欧氏backbone
        hyperbolic_classifier=True,    # 双曲分类头
        use_group_norm=True,
        time_enc=True
    )
    x = torch.randn(2, 4, 2048)  # 小点数加速测试
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == (2, 101), f"Expected (2, 101), got {out.shape}"
    assert torch.isfinite(out).all(), "Output has NaN/Inf"

    # 反向
    loss = out.sum()
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")

    # 参数量
    total = sum(p.numel() for p in model.parameters())
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"  Total params: {total:,}")
    print(f"  Classifier params: {clf_params:,}")
    print("  PASS")


def test_param_count_comparison():
    """对比双曲分类头 vs 普通分类头参数量"""
    print("\n=== Test 5: Param count comparison ===")
    from lr_hypernet import LRHyperNet
    model_hyp = LRHyperNet(num_classes=101, hyperbolic_type="euclidean",
                           hyperbolic_classifier=True, use_group_norm=True)
    model_std = LRHyperNet(num_classes=101, hyperbolic_type="euclidean",
                           hyperbolic_classifier=False, use_group_norm=True)
    p_hyp = sum(p.numel() for p in model_hyp.parameters())
    p_std = sum(p.numel() for p in model_std.parameters())
    print(f"  Hyperbolic classifier: {p_hyp:,}")
    print(f"  Standard classifier:   {p_std:,}")
    print(f"  Diff: {p_hyp - p_std:,} ({(p_hyp-p_std)/p_std*100:.2f}%)")
    print("  PASS")


if __name__ == "__main__":
    print("Hyperbolic Prototype Classifier CPU Verification")
    print("=" * 50)
    test_classifier_forward()
    test_classifier_backward()
    test_lorentz_functions()
    test_full_model()
    test_param_count_comparison()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
