"""learned scorer 地基：可微 top-k 数值原型（ST_k，NeurIPS 2024，Codex 复审版）。

Codex 复审（2026-10-24）三处口径修正：
1. λ 不是可训练参数——是 forward 内非可微地取 kthvalue(s).detach()，每次重算，
   天然保证 Σw≈k。若照"可训练 λ"天真实现会漂移（潜在 P0）。
2. δ 尺度必须与 score 尺度匹配（[0,1] 分数需 δ~1e-5，1e-4 会打脸质量集中断言）；
   实际 scorer 输出要先归一化（tanh/标准化到有界区间），再用固定 δ 调度（大→小）。
3. Σw 是 "≈k"（λ 取第 k 大时，第 k 大元素落在阈值处 w=0.5，Σw≈k−0.5），不是精确 k。

ST_k 核心：w_i = SReLU'(s_i − λ) = ½[1 + (s_i−λ)/√((s_i−λ)²+δ²)]

用法（3090）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_soft_topk.py
"""
import torch


def stk_soft_topk(scores: torch.Tensor, k: int, delta: float = 1e-5) -> torch.Tensor:
    """ST_k 软权重。λ 非可微地取第 k 大（detach），每次 forward 重算。"""
    lam = scores.detach().kthvalue(scores.numel() - k + 1).values
    x = scores - lam
    return 0.5 * (1 + x / torch.sqrt(x * x + delta * delta))


n, k = 10240, 16
torch.manual_seed(0)
# 演示「归一化分数」：scorer 输出标准化到 mean 0 / std 1（scale 稳定）
raw = torch.randn(n, requires_grad=True)
scores = (raw - raw.mean()) / (raw.std() + 1e-8)

# δ 用与归一化后分数尺度匹配的固定值（第 16/17 大间隔 ~1e-4 量级 → δ=1e-5）
w = stk_soft_topk(scores, k, delta=1e-5)

# 1. Σw ≈ k（不是精确 k；第 k 大元素 w=0.5 → Σw≈k−0.5）
s = w.sum().item()
print(f"Σw = {s:.3f} (期望 ≈ {k}，Codex：实际 ≈k−0.5)")
assert abs(s - k) < 1.0, f"Σw 偏离 k 太多: {s}"

# 2. 质量集中：非 top-k 权重接近 0（δ=1e-5 下通过；δ=1e-4 会打脸）
topk = torch.topk(scores, k).indices
mask = torch.ones(n, dtype=torch.bool)
mask[topk] = False
print(f"非 top-{k} 权重 max = {w[mask].max().item():.4f} (期望 < 0.1)")
assert w[mask].max() < 0.1, "δ 太大，非 top-k 权重没压下去（需更小 δ）"

# 3. 梯度非负（SReLU' 单调 → score 高 → 权重高），且集中在阈值附近
w.sum().backward()
grad = raw.grad
print(f"梯度范围 [{grad.min().item():.4f}, {grad.max().item():.4f}]（应全 ≥0，稀疏集中在阈值附近）")
assert grad.min() >= -1e-6, "梯度应非负"

print("\nST_k 三点验证通过 ✓（λ=kthvalue detach，分数归一化 + δ 匹配尺度）")
