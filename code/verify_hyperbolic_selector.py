"""验证双曲距离选点 vs 欧氏距离选点的差异。

测试内容：
1. 两种选点方式选出的远程邻居索引是否不同（验证双曲距离确实改变了选点结果）
2. 双曲选点模型能否正常前向+反向传播
3. 参数量对比（双曲选点不增加参数，曲率是 buffer）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import torch
from lr_hypernet import LRHyperNet


def test_selector_difference():
    """验证双曲选点和欧氏选点选出的索引不同。"""
    print("=== 测试1：双曲选点 vs 欧氏选点 索引差异 ===")
    from models.local_remote_grouper import RemoteSelector

    torch.manual_seed(42)
    B, S, N, C = 2, 64, 512, 64
    centroid = torch.randn(B, S, C)
    all_feat = torch.randn(B, N, C)
    local_idx = torch.randint(0, N, (B, S, 24))

    # 欧氏选点
    selector_euc = RemoteSelector(remote_g=16, use_hyperbolic=False)
    idx_euc = selector_euc(centroid, all_feat, local_idx)

    # 双曲选点（lorentz）
    selector_hyp = RemoteSelector(remote_g=16, use_hyperbolic=True,
                                   curvature_init=0.4, hyperbolic_type="lorentz")
    idx_hyp = selector_hyp(centroid, all_feat, local_idx)

    # 计算重合率
    overlap = 0
    total = B * S * 16
    for b in range(B):
        for s in range(S):
            set_euc = set(idx_euc[b, s].tolist())
            set_hyp = set(idx_hyp[b, s].tolist())
            overlap += len(set_euc & set_hyp)
    overlap_rate = overlap / total
    print(f"  欧氏选点索引范围: [{idx_euc.min()}, {idx_euc.max()}]")
    print(f"  双曲选点索引范围: [{idx_hyp.min()}, {idx_hyp.max()}]")
    print(f"  重合率: {overlap_rate:.2%} ({overlap}/{total})")
    assert overlap_rate < 0.95, "双曲选点与欧氏选点几乎完全相同，可能有bug"
    print("  ✅ 双曲距离确实改变了选点结果")


def test_model_forward():
    """测试双曲选点模型能否正常前向+反向传播。"""
    print("\n=== 测试2：双曲选点模型前向+反向传播 ===")
    for cfg_name, cfg in [
        ("v3.1 + 双曲选点 (lorentz)", dict(
            time_enc=True, use_global_anchor=True,
            hyperbolic_type="lorentz", use_hyperbolic_selector=True)),
        ("v3.1 + 双曲选点 (poincare)", dict(
            time_enc=True, use_global_anchor=True,
            hyperbolic_type="poincare", use_hyperbolic_selector=True)),
    ]:
        print(f"\n  --- {cfg_name} ---")
        model = LRHyperNet(num_classes=101, **cfg)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,}")

        # 前向
        x = torch.randn(2, 4, 1024)
        with torch.no_grad():
            out = model(x)
        print(f"  前向输出形状: {out.shape}, finite: {torch.isfinite(out).all().item()}")
        assert out.shape == (2, 101)

        # 反向
        model.train()
        x = torch.randn(2, 4, 1024)
        label = torch.randint(0, 101, (2,))
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, label)
        loss.backward()
        bad_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f"  反向 loss: {loss.item():.4f}, 非有限梯度: {bad_grads if bad_grads else '无'}")
        assert not bad_grads
        print(f"  ✅ {cfg_name} 前向+反向通过")


def test_selector_no_extra_params():
    """验证双曲选点不增加可训练参数（曲率是 buffer）。"""
    print("\n=== 测试3：参数量对比 ===")
    model_euc = LRHyperNet(num_classes=101, time_enc=True,
                           use_global_anchor=True, hyperbolic_type="lorentz",
                           use_hyperbolic_selector=False)
    model_hyp = LRHyperNet(num_classes=101, time_enc=True,
                           use_global_anchor=True, hyperbolic_type="lorentz",
                           use_hyperbolic_selector=True)
    params_euc = sum(p.numel() for p in model_euc.parameters())
    params_hyp = sum(p.numel() for p in model_hyp.parameters())
    print(f"  欧氏选点参数量: {params_euc:,}")
    print(f"  双曲选点参数量: {params_hyp:,}")
    print(f"  差异: {params_hyp - params_euc:,} (应为0，曲率是buffer不是parameter)")
    assert params_euc == params_hyp, "双曲选点不应增加可训练参数"
    print("  ✅ 双曲选点不增加可训练参数")


if __name__ == "__main__":
    test_selector_difference()
    test_model_forward()
    test_selector_no_extra_params()
    print("\n" + "=" * 50)
    print("全部测试通过！双曲距离选点功能正常。")
