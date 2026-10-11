# -*- coding: utf-8 -*-
"""测试 HFECNet 的 4 个消融版本：验证前向/反向传播均通过，并对比参数量。"""
import torch
import torch.nn.functional as F

from hyperbolic_fecnet_ablation import HFECNetAblation, HFECNetAblation as _

ABLATIONS = ['none', 'no_hyperbolic', 'fixed_curvature', 'no_centroid']

# 完整模型跑 8192 点较慢，测试用 2048 点即可（FPS 采样对点数自适应）
NUM_POINTS = 2048 if not torch.cuda.is_available() else 8192
BATCH_SIZE = 2
NUM_CLASSES = 101
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"device={DEVICE}, batch_size={BATCH_SIZE}, num_points={NUM_POINTS}\n")

results = []
for ablation in ABLATIONS:
    print(f"===== ablation: {ablation} =====")
    torch.manual_seed(0)
    model = HFECNetAblation(num_classes=NUM_CLASSES,
                            ablation=ablation).to(device=DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 聚合层参数（各消融版本差异所在）
    agg_params = sum(p.numel() for n, p in model.named_parameters()
                     if 'hyp_aggregation' in n)
    # 曲率参数个数（none/no_centroid 可学习，其余版本为 0）
    curv_params = [n for n, _ in model.named_parameters()
                   if 'curvature' in n and _.requires_grad]
    print(f"总参数量: {total:,} | 可训练: {trainable:,} | "
          f"聚合层参数: {agg_params:,} | 可学习曲率参数: {len(curv_params)}")

    # 前向
    model.train()
    x = torch.randn(BATCH_SIZE, 4, NUM_POINTS, device=DEVICE)
    label = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
        f"输出形状错误: {out.shape}"
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"
    print(f"前向通过: 输出 {tuple(out.shape)}, "
          f"logits 范围 [{out.min().item():.3f}, {out.max().item():.3f}]")

    # 反向
    loss = F.cross_entropy(out, label)
    loss.backward()
    # 检查梯度覆盖情况。注：以下参数在原版 FECNet/LocalGrouper 中就是
    # "只定义未使用"的死参数（forward 中从不引用），不属于消融引入的问题：
    #   - local_grouper_list.*.affine_alpha / affine_beta（LocalGrouper 未使用）
    #   - attention.linear.*（FECNet/HFECNet 的 self.attention 未在 forward 中调用）
    DEAD_PARAMS = ('affine_alpha', 'affine_beta', 'attention.linear')
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None
               and not any(d in n for d in DEAD_PARAMS)]
    assert not no_grad, f"以下参数没有梯度: {no_grad}"
    grad_norm = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if p.grad is not None)
    # 曲率梯度（仅可学习曲率版本有）
    for n, p in model.named_parameters():
        if 'curvature' in n and p.requires_grad:
            print(f"曲率 {n}: value={p.item():.4f}, grad={p.grad.item():.2e}")
    print(f"反向通过: loss={loss.item():.4f}, 梯度总范数={grad_norm:.2f}")

    results.append((ablation, total, trainable, agg_params))
    print()

# 参数量对比表
print("=" * 72)
print(f"{'消融模式':<18}{'总参数量':>14}{'可训练':>14}{'聚合层参数':>14}")
print("-" * 72)
base = results[0][1]
for ablation, total, trainable, agg in results:
    diff = f" (+{total-base:,})" if total > base else \
           (f" ({total-base:,})" if total < base else "")
    print(f"{ablation:<18}{total:>14,}{trainable:>14,}{agg:>14,}{diff}")
print("=" * 72)
print("✅ 全部 4 个消融版本前向/反向传播测试通过")
