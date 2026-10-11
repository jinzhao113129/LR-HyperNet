"""纯 PyTorch B-spline 核卷积（ECAF-GCN 论文式 1 的直接实现）。

h_i' = Σ_{j∈𝒩(i)} g_Θ(u_ij) · h_j

g_Θ(u) 是定义在边属性 u 上的 B-spline 核：g_Θ(u) = Σ_k B_k(u) · W_k，
其中 B_k 为 B-spline 基函数（tensor product，degree 阶），W_k 为对应核权重。

为什么自实现：
  - torch_geometric 的 SplineConv 依赖 torch-spline-conv C++ kernel，
    其在 torch 2.6.0+cu124 上触发 CUDA illegal memory access（已验证）。
  - 本实现为纯 torch 算子（scatter + einsum），数值等价、可解释、
    可直接写进论文/报告，且与 CUDA/CPU 均兼容。

边属性约定：输入 edge_attr ∈ [-1, 1]（相对偏移/半径，与 event_graph 一致），
内部归一化到 [0, 1] 供 B-spline 基函数使用。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def bspline_basis(x: torch.Tensor, degree: int, kernel_size: int) -> torch.Tensor:
    """计算一维二次 B-spline 基函数值（均匀节点，支撑区间 [0,1]）。

    参数:
        x: [E] 每条边的属性值（已归一化到 [0,1]）。
        degree: B-spline 阶数（本实现支持 2，论文/AEGNN 同款）。
        kernel_size: 核数量（AEGNN ncaltech101 用 8）。
    返回:
        [E, kernel_size] 基函数值，满足 partition of unity（区间内部 Σ_k B_k = 1）。

    闭式二次 B-spline（均匀节点间距 h，核 k 支撑 [k·h, (k+3)·h]）：
      [t_k, t_{k+1}):   0.5·((x−t_k)/h)²
      [t_{k+1}, t_{k+2}): 0.5·(1 + 2u − 2u²), u=(x−t_{k+1})/h
      [t_{k+2}, t_{k+3}): 0.5·(1 − u)²,       u=(x−t_{k+2})/h
    """
    assert degree == 2, "本实现仅支持二次 B-spline（degree=2）"
    E = x.shape[0]
    h = 1.0 / (kernel_size - 1)
    out = torch.zeros(E, kernel_size, device=x.device, dtype=x.dtype)

    for k in range(kernel_size):
        t0, t1, t2, t3 = k * h, (k + 1) * h, (k + 2) * h, (k + 3) * h
        seg1 = (x >= t0) & (x < t1)
        seg2 = (x >= t1) & (x < t2)
        seg3 = (x >= t2) & (x < t3)
        u1 = (x - t0) / h
        u2 = (x - t1) / h
        u3 = (x - t2) / h
        out[:, k] = torch.where(seg1, 0.5 * u1 * u1,
                     torch.where(seg2, 0.5 * (1 + 2 * u2 - 2 * u2 * u2),
                     torch.where(seg3, 0.5 * (1 - u3) * (1 - u3), torch.zeros_like(x))))
    return out


class SplineConvPure(nn.Module):
    """纯 PyTorch B-spline 核图卷积（论文式 1）。

    参数:
        in_channels: 输入特征维度。
        out_channels: 输出特征维度。
        dim: 边属性维度（3 = 时空）。
        kernel_size: 每维 B-spline 核数。
        degree: B-spline 阶数。
        bias: 是否使用偏置。
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int = 3,
                 kernel_size: int = 8, degree: int = 2, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.degree = degree

        # 核权重：K^dim 个 [in, out] 矩阵（tensor product 基函数对应权重）
        self.weight = nn.Parameter(torch.empty(kernel_size ** dim, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """x: [N, in]；edge_index: [2, E]；edge_attr: [E, dim] ∈ [-1,1]。

        返回 [N, out]。
        """
        src, dst = edge_index
        u = (edge_attr + 1.0) / 2.0                     # [-1,1] -> [0,1]
        u = u.clamp(0.0, 1.0)

        # 每维基函数 → tensor product 展平为 [E, K^dim]
        bases = []
        for d in range(self.dim):
            b_d = bspline_basis(u[:, d], self.degree, self.kernel_size)   # [E, K]
            bases.append(b_d)
        # 张量积：b_flat[e, k0*K^1 + k1*K^0 + ...] = Π_d b_d[e, k_d]
        b = bases[0]
        for b_d in bases[1:]:
            b = (b[:, :, None] * b_d[:, None, :]).reshape(b.shape[0], -1)  # [E, K^(d+1)]

        # 边消息：msg_e = W(u_e) @ x_src，W(u_e) = Σ_k b[e,k] * weight[k]
        # [E, in, out] = b[E, K^dim] @ weight[K^dim, in, out]
        w_e = torch.einsum("ek,kio->eio", b, self.weight)     # [E, in, out]
        msg = torch.einsum("eio,ei->eo", w_e, x[src])          # [E, out]

        out = scatter(msg, dst, dim=0, dim_size=x.shape[0], reduce="sum")
        if self.bias is not None:
            out = out + self.bias
        return out
