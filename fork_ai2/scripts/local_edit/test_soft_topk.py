"""learned scorer 地基：可微 top-k 数值原型（ST_k，NeurIPS 2024）。

排坑记录（2026-10-24）：
- sigmoid top-k：Σw 爆到 711；sequential softmax：质量不集中。
- 本地 δ sweep 发现：可微 top-k 的 δ/τ 必须与 score 尺度匹配（[0,1] 需 δ~1e-5）。
- ST_k 的独有优势：阈值 λ 是可训练参数，自动收敛到第 k 大元素，无需手动调温度。

ST_k 核心（Ogryczak-Tamir Eq.1 + SReLU）：
    w_i = SReLU'(s_i - λ) = 0.5 * (1 + (s_i-λ)/sqrt((s_i-λ)^2 + δ^2))
    λ 初始化为第 k 大元素，随训练自动收敛；δ 退火（大→小）。

验证三点（Codex 要求）：Σw≈k、质量集中、梯度方向（w 随 score 单调）。

用法（3090）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_soft_topk.py
"""
import torch


def stk_soft_topk(scores: torch.Tensor, k: int, delta: float = 1e-4,
                  lam: torch.Tensor = None) -> torch.Tensor:
    """ST_k 软权重：w_i = SReLU'(s_i - λ)。

    scores: [n]（scorer 打分，可含梯度）
    lam: 阈值（可训练参数，默认初始化为第 k 大元素）
    delta: 平滑系数（需与 score 尺度匹配，退火）
    """
    if lam is None:
        lam = scores.detach().kthvalue(scores.numel() - k + 1).values
    x = scores - lam
    return 0.5 * (1 + x / torch.sqrt(x * x + delta * delta))


n, k = 10240, 16
torch.manual_seed(0)
scores = torch.rand(n, requires_grad=True)

# δ 需匹配 score 尺度（[0,1] 用 1e-4）
w = stk_soft_topk(scores, k, delta=1e-4)

# 1. Σw ≈ k
print(f"Σw = {w.sum().item():.3f} (期望 ≈ {k})")
assert abs(w.sum().item() - k) < k * 0.15, "Σw 偏离 k 太多"

# 2. 质量集中：top-k 权重接近 1，非 top-k 接近 0
topk = torch.topk(scores, k).indices
mask = torch.ones(n, dtype=torch.bool)
mask[topk] = False
print(f"top-{k} 权重 min = {w[topk].min().item():.3f}（第 k 大在阈值上≈0.5 属正常）")
print(f"非 top-k 权重 max = {w[mask].max().item():.3f}（期望接近 0）")
assert w[mask].max() < 0.1, "非 top-k 权重应接近 0"

# 3. 梯度方向：w 随 score 单调（SReLU' 单调递增 → ∂w/∂score > 0）
w.sum().backward()
grad = scores.grad
print(f"梯度范围 [{grad.min().item():.4f}, {grad.max().item():.4f}]（应全 ≥ 0）")
assert grad.min() >= 0, "梯度应为非负（score 高 → 权重高）"

print("\nST_k 三点验证通过 ✓（λ 可训练是相对 STE 的核心优势）")
