"""Poincaré 双曲几何运算（ECAF-GCN 论文复现）。

按论文逐式实现：
  附录 E 式 16：双曲距离   d(x,y) = (1/√c) · arccosh(1 + 2c||x−y||² / ((1−c||x||²)(1−c||y||²)))
  附录 E 式 17：Möbius 加法 x⊕_c y
  附录 E 式 18：Möbius 矩阵-向量乘 M ⊗_c x = exp_o^c(M · log_o^c(x))
  Section 3 式 2：指数映射   exp_o^c(v) = o ⊕_c (tanh(√c·||v||/2) · v/(√c·||v||))
  Section 3 式 3-5：双曲 GCN 层（Möbius 线性 + 切空间聚合 + 切空间激活）

所有运算以原点 o 为锚点（论文式 2/18 均为原点锚定）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_to_disk(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """把点投影回 Poincaré 球内部：||x|| < 1/√c − eps，防止数值发散。"""
    max_norm = (1.0 / torch.sqrt(c)) - eps
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    cond = norm > max_norm
    scale = torch.where(cond, max_norm / norm, torch.ones_like(norm))
    return x * scale


def poincare_log_origin(y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对数映射（原点锚定）：log_o^c(y) = (2/√c) · arctanh(√c·||y||) · y/||y||。

    o=0 时的闭式解（Ganea et al. 2018）。与式 18 配套使用。
    数值稳定性：||y||→0 时返回 0 向量（log_o(o)=0）。
    """
    norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(eps)
    arg = (torch.sqrt(c) * norm).clamp(max=1.0 - eps)
    coef = (2.0 / torch.sqrt(c)) * torch.arctanh(arg)
    return torch.where(norm <= eps, torch.zeros_like(y), coef * (y / norm))


def poincare_exp_origin(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """指数映射（原点锚定）：exp_o^c(v) = tanh(√c·||v||/2) · v/(√c·||v||)（式 2，o=0 简化）。"""
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    coef = torch.tanh(torch.sqrt(c) * norm / 2.0) / (torch.sqrt(c) * norm)
    out = coef * v
    return torch.where(norm <= eps, torch.zeros_like(out), clip_to_disk(out, c))


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Möbius 加法（式 17）。

    x⊕_c y = ((1+2c⟨x,y⟩+c||y||²)x + (1−c||x||²)y) / (1+2c⟨x,y⟩+c²||x||²||y||²)
    """
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(eps)
    return clip_to_disk(num / den, c)


def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Möbius 矩阵-向量乘（式 18）：M ⊗_c x = exp_o^c(M · log_o^c(x))。

    M: [d_out, d_in]；x: [..., d_in]；返回 [..., d_out]（双曲空间）。
    """
    log_x = poincare_log_origin(x, c)          # → 切空间
    lin = log_x @ M.t()                         # 线性变换（欧氏）
    return poincare_exp_origin(lin, c)          # → 双曲空间


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """双曲距离（式 16）。

    d(x,y) = (1/√c) · arccosh(1 + 2c||x−y||² / ((1−c||x||²)(1−c||y||²)))
    数值稳定：对 arccosh 用 log 等价形式 arccosh(z) = log(z + √(z²−1))。
    """
    x_norm_sq = (x * x).sum(-1).clamp(max=(1.0 / c) - eps)
    y_norm_sq = (y * y).sum(-1).clamp(max=(1.0 / c) - eps)
    diff_sq = ((x - y) ** 2).sum(-1)
    z = 1 + 2 * c * diff_sq / ((1 - c * x_norm_sq) * (1 - c * y_norm_sq)).clamp_min(eps)
    # arccosh(z) = log(z + sqrt(z^2 - 1))，z ≥ 1
    z = z.clamp_min(1.0 + eps)
    arccosh_z = torch.log(z + torch.sqrt((z - 1.0) * (z + 1.0)))
    return (1.0 / torch.sqrt(c)) * arccosh_z


class HyperbolicGraphConv(nn.Module):
    """双曲 GCN 层（论文式 3-5，三步骤消息传递）。

    式 3：F̃_i = (Γ ⊗_c F_i) ⊕_c b            —— Möbius 线性变换 + 偏置
    式 4：F̃_i = exp_o( Σ_j A_ij · log_o(F̃_j) ) —— 切空间邻域加权聚合
    式 5：F_i' = exp_o( φ( log_o(F̃_i) ) )     —— 切空间非线性激活后投影回双曲

    参数：
        in_dim: 输入维度；out_dim: 输出维度；c: 可学习曲率（标量或 [out_dim]）。
    注：聚合权重 A_ij 采用均值（1/|𝒩(i)|）【假设，论文未给归一化方式】；
        式 3 的偏置 b 定义在原点切空间，先经指数映射后做 Möbius 加法。
    """

    def __init__(self, in_dim: int, out_dim: int, curvature_init: float = 0.4,
                 bias: bool = True, dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.curvature = nn.Parameter(torch.tensor(curvature_init))  # 可学习曲率（论文：init 0.4）
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = {"relu": F.relu, "elu": F.elu, "tanh": torch.tanh, "identity": lambda x: x}[act]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _curvature(self) -> torch.Tensor:
        # 曲率需为正：论文 c ∈ {0.2,0.4,0.6,0.8}，用 softplus 保证正值。
        return F.softplus(self.curvature) + 0.05

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim]（双曲空间点）；edge_index: [2, E]；返回 [N, out_dim]（双曲空间）。"""
        c = self._curvature()

        # 式 3：Möbius 线性变换 + 偏置（偏置经指数映射进双曲空间再 Möbius 加）
        h = mobius_matvec(self.weight, x, c)                    # Γ ⊗_c x
        if self.bias is not None:
            b_h = poincare_exp_origin(self.bias.unsqueeze(0), c)  # b ∈ T_o → 双曲空间
            h = mobius_add(h, b_h.expand_as(h), c)               # ⊕_c b

        # 式 4：邻居特征 log 到切空间 → 均值聚合（A_ij = 1/|𝒩(i)|）→ exp 回双曲
        src, dst = edge_index
        h_log = poincare_log_origin(h, c)                        # [N, d] 切空间
        agg = torch.zeros_like(h_log)
        agg.index_add_(0, dst, h_log[src])                       # Σ_j log(F̃_j)
        deg = torch.zeros(x.shape[0], 1, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=x.dtype).unsqueeze(1))  # |𝒩(i)|
        agg = agg / deg.clamp_min(1.0)                           # 均值
        h_agg = poincare_exp_origin(agg, c)                      # 回双曲空间

        # 式 5：切空间激活后再投影回双曲
        h_log2 = poincare_log_origin(h_agg, c)
        h_out = poincare_exp_origin(self.act(h_log2), c)
        return self.dropout(h_out)
