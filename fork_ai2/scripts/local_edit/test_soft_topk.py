"""learned scorer 地基：可微 top-k 数值原型（STE straight-through）。

背景（2026-10-24 排坑记录）：
- sigmoid top-k：Σw 爆到 711（τ 与 score 尺度不匹配）
- sequential softmax：Σw=16 精确但 top-16 权重只 0.017（质量不集中）
- 正确方案需 NeuralSort/SOFT/ST_k 等专门设计；ReparamSubset 的 RelaxedTopK
  本身无公式，引用 Grover 2019 的松弛置换矩阵。

本原型用 STE：前向硬 top-k（精确 one-hot，Σw=k、质量集中天然精确），
反向用 softmax 软权重梯度（方向对）。验证三点（Codex 要求）：
1. 前向 Σw = k 精确
2. 前向质量集中在 top-k（非 top-k = 0）
3. 反向梯度 ∂Σw/∂score 对 top-k 为正（score 高 → 权重高）

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_soft_topk.py
"""
import torch


def ste_topk(scores: torch.Tensor, k: int, tau: float = 0.1) -> torch.Tensor:
    """STE 可微 top-k：前向硬 one-hot，反向 softmax 梯度。

    scores: [n]（scorer 打分，可含梯度）
    返回: [n] 软权重，前向值 = top-k one-hot（Σ=k），反向梯度 = softmax 的梯度
    """
    topk = torch.topk(scores, k).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(0, topk, 1.0)  # 硬 one-hot（前向）
    soft = torch.softmax(scores / tau, dim=0)  # 软权重（反向梯度来源）
    return hard - soft.detach() + soft  # STE：前向 hard，反向 soft 梯度


n, k = 10240, 16
torch.manual_seed(0)
scores = torch.rand(n, requires_grad=True)

w = ste_topk(scores, k, tau=0.1)

# 1. 前向 Σw = k 精确
print(f"Σw = {w.sum().item():.6f} (期望 = {k} 精确)")
assert abs(w.sum().item() - k) < 1e-5, "前向 Σw 不等于 k"

# 2. 前向质量集中：top-k 全为 1，非 top-k 全为 0
topk = torch.topk(scores, k).indices
mask = torch.ones(n, dtype=torch.bool)
mask[topk] = False
assert (w[topk] > 0.99).all(), "top-k 权重应=1"
assert w[mask].abs().max() < 1e-5, "非 top-k 权重应=0"
print(f"前向：top-{k} 权重=1，其余=0（质量集中精确）✓")

# 3. 反向梯度：∂Σw/∂score 对 top-k 为正（softmax 梯度 w(1-w)/tau > 0）
w.sum().backward()
grad = scores.grad
grad_topk = grad[topk].min().item()
grad_rest = grad[mask].max().item()
print(f"反向梯度：top-k min={grad_topk:.4f} (>0)，非 top-k max={grad_rest:.4f}")
assert grad_topk > 0, "反向梯度方向错（top-k 应正梯度）"

print("\nSTE 可微 top-k 三点验证通过 ✓")
