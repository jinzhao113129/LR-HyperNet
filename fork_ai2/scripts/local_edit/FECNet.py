import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LocalGrouper, index_points, square_distance, knn_chunk_fallback
from typing import Any, Callable, List, Optional, Union
from numbers import Number
import numpy as np

try:  # 兼容两种导入上下文：从项目根目录包导入 / models 目录在 sys.path 中
    from local_remote_grouper import (KDACausalEncoder, HyperbolicPrototypeClassifier,
                                      lorentz_exp_origin, lorentz_log_origin, lorentz_distance,
                                      poincare_exp_origin, poincare_log_origin, clip_to_disk)
except ImportError:  # pragma: no cover
    from models.local_remote_grouper import (KDACausalEncoder, HyperbolicPrototypeClassifier,
                                             lorentz_exp_origin, lorentz_log_origin,
                                             lorentz_distance,
                                             poincare_exp_origin, poincare_log_origin,
                                             clip_to_disk)


def _largest_group_count(channels: int, max_groups: int = 32) -> int:
    """取不超过 max_groups 且能整除 channels 的最大分组数。

    GroupNorm 要求 channels % num_groups == 0，而本网络通道数（132/268/540/66/134/270
    等）大多不是 32 的倍数，故从 min(32, C) 向下搜索第一个合法因子。
    """
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


def make_norm1d(channels: int, use_group_norm: bool = False) -> nn.Module:
    """归一化层工厂：默认 BatchNorm1d，--group_norm 时换 GroupNorm。

    b8 小 batch 下 BatchNorm 的批统计量（均值/方差）仅由 8 个样本估计，噪声极大，
    是 b8 变体严重欠拟合的嫌疑之一；GroupNorm 按 channel 分组归一化，统计量
    与 batch size 无关，小 batch 下行为稳定。
    """
    if use_group_norm:
        return nn.GroupNorm(_largest_group_count(channels), channels)
    return nn.BatchNorm1d(channels)


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True,
                 use_group_norm: bool = False):
        super(Linear1Layer, self).__init__()
        self.act = nn.GELU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            make_norm1d(out_channels, use_group_norm),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True,
                 use_group_norm: bool = False):
        super(Linear2Layer, self).__init__()
        self.act = nn.GELU()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels / 2),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            make_norm1d(int(in_channels / 2), use_group_norm),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=int(in_channels / 2), out_channels=in_channels,
                      kernel_size=kernel_size, bias=bias),
            make_norm1d(in_channels, use_group_norm)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class fftlayer(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.complex_weight = weight

    def forward(self, x):
        x = torch.fft.rfft(x, dim=(1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, dim=(1), norm='ortho')
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = self.linear(lstm_output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs


# ============================================================================
# 组件版本（消融实验）：全部默认关闭，关闭时与原始 FECNet 逐字节等价
# 傅里叶聚合（fft_sptaial/fft_temporal）不做任何改动
# ============================================================================

def time_freq_encode(xyz: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """多频 sin/cos 时间编码（版本1 --time_enc）。

    对 t 通道（xyz[..., 0]，已归一化到 [0,1]）以频率 2^0..2^{num_freqs-1}
    做 sin/cos 编码，输出 [B, N, 2*num_freqs]（默认 8 维），替代原始 t
    通道拼进 embed 输入。

    Args:
        xyz: [B, N, 4] 事件坐标 (t, x, y, p)
        num_freqs: 频率个数（频率 = 2^0 .. 2^{num_freqs-1}）
    Returns:
        [B, N, 2*num_freqs] 时间编码
    """
    t = xyz[..., 0:1]  # [B, N, 1]
    parts = []
    for i in range(num_freqs):
        freq = 2.0 ** i
        parts.append(torch.sin(2 * math.pi * freq * t))
        parts.append(torch.cos(2 * math.pi * freq * t))
    return torch.cat(parts, dim=-1)  # [B, N, 2*num_freqs]


class KDAPointEncoder(nn.Module):
    """逐点 KDA 时间编码（版本2 --kda，time_enc 的替代）。

    embed 之前对每个事件取其 (t,x,y) 空间 K=24 近邻（含自身），组内按
    时间戳升序做 gated delta rule 因果递推（复用已验证的
    KDACausalEncoder），取自身时间位置的状态读出作为该事件的
    state_dim 维时间编码，替代原始 t 通道。

    输出维度 = state_dim（默认 16），embed 输入变成 (x,y,p + 16) = 19 维。
    """

    def __init__(self, kneighbors: int = 24, state_dim: int = 16, chunk: int = 1024):
        super().__init__()
        self.kneighbors = kneighbors
        self.state_dim = state_dim
        self.chunk = chunk
        # 原始 4 维 xytp → state_dim，使 KDACausalEncoder 输出即 state_dim 维
        self.in_proj = nn.Linear(4, state_dim)
        self.kda = KDACausalEncoder(channel=state_dim, state_dim=state_dim)

    def _knn_idx(self, xyz: torch.Tensor) -> torch.Tensor:
        """(t,x,y) 空间分块 KNN（含自身，自身距离 0 必入选）。

        完整 [B, N, N] 距离矩阵在 N=10240 时过大，按中心点分块：每块中心
        对全部 N 个点取 top-K（候选集是全集，故结果即精确全局 KNN），
        沿中心维拼回 [B, N, K]。
        """
        N = xyz.shape[1]
        K = self.kneighbors
        topk_list = []
        for start in range(0, N, self.chunk):
            end = min(start + self.chunk, N)
            chunk_dist = square_distance(xyz[:, start:end, :], xyz)  # [B, c, N]
            topk_list.append(chunk_dist.topk(min(K, N), dim=-1, largest=False)[1])
        idx = torch.cat(topk_list, dim=1)                # [B, N, K]
        return idx.sort(dim=-1)[0]

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 4] 事件坐标 (t, x, y, p)
        Returns:
            [B, N, state_dim] 每个事件的 KDA 时间编码
        """
        B, N, _ = xyz.shape
        idx = self._knn_idx(xyz[:, :, 0:3])              # [B, N, K]（t,x,y 空间）
        grouped = index_points(xyz, idx)                 # [B, N, K, 4]
        ts = grouped[..., 0:1]                           # [B, N, K, 1]
        # 自身在组内的位置（KNN 含自身，距离 0 必入选；取首个匹配兜底）
        self_in = (idx == torch.arange(N, device=xyz.device).view(1, N, 1))  # [B, N, K]
        # 修复F：防御断言——若某点有 >K 个零距离重合点且自身落在 tie-break 之外，
        # self_in 全 False，argmax=0 会静默取错位置；正常数据（含自身的精确 KNN）
        # 必然命中，断言不触发
        assert self_in.any(dim=-1).all(), "KDA 邻居缺失自身索引"
        self_pos = self_in.float().argmax(dim=-1)        # [B, N]
        # KDACausalEncoder 内部按 ts 升序排序后递推，输出按时间序排列；
        # 自身的时间序位置 = order[j] == self_pos 的那个 j（同一 argsort 调用，确定性一致）
        order = ts[..., 0].argsort(dim=-1)               # [B, N, K]
        sorted_pos = (order == self_pos.unsqueeze(-1)).float().argmax(dim=-1)  # [B, N]
        feats = self.in_proj(grouped)                    # [B, N, K, D]
        out = self.kda(feats, ts)                        # [B, N, K, D]（时间序）
        g = sorted_pos.view(B, N, 1, 1).expand(B, N, 1, self.state_dim)
        return out.gather(2, g).squeeze(2)               # [B, N, D]


class RP1SpatialUniformGrouper(LocalGrouper):
    """RP1 spatial-uniform 选点版 LocalGrouper（版本3 --rp1_selector）。

    只替换 FPS 采样：group center 改为在 x,y 空间均匀网格采样（RP1 论文
    最好的 cheap selector，参考 local_remote_grouper.RemoteSelector 的
    _spatial_uniform_select）。x,y ∈ [0,1] 均分为 grid_size×grid_size 网格
    （grid_size = ceil(√S)，如 S=1024 → 32×32），每格随机取 1 点，按随机
    网格访问序凑够 S 个 center，非空网格不足时随机补齐。S 仍等于
    group_number[i]（下游 fft_temporal 权重形状依赖 S，不能变）。

    其余（KNN、归一化、特征拼接）与原 LocalGrouper 逐行一致。
    """

    @staticmethod
    def _spatial_uniform_idx(xytp: torch.Tensor, S: int) -> torch.Tensor:
        """x,y 空间均匀网格采样 S 个点索引。

        Args:
            xytp: [B, N, 4]（t,x,y,p），x,y 已归一化到 [0,1]
            S: 需要的 center 数（= group_number[i]）
        Returns:
            [B, S] 选中点索引（未排序，调用方排序）
        """
        B, N, _ = xytp.shape
        grid_size = math.ceil(math.sqrt(S))
        n_cells = grid_size * grid_size
        eps = 1e-6
        # x,y ∈ [0,1]（布局 (t,x,y,p)：x=通道1, y=通道2），clamp 兜底防越界
        cell_x = (xytp[..., 1].clamp(0.0, 1.0 - eps) * grid_size).long()
        cell_y = (xytp[..., 2].clamp(0.0, 1.0 - eps) * grid_size).long()
        cell = cell_x * grid_size + cell_y               # [B, N]
        rand = torch.rand(B, N, device=xytp.device)
        # 优先级编码（与 _spatial_uniform_select 同一技巧）：
        #   网格赢家 ∈ (1, 2]，网格访问序越早优先级越高；其余 = rand ∈ [0,1)
        #   作非空网格不足 S 时的随机补齐池
        priority = rand.clone()
        cell_order = torch.randperm(n_cells, device=xytp.device)
        for seq, j in enumerate(cell_order.tolist()):
            in_cell = (cell == j)                        # [B, N]
            if not in_cell.any():
                continue
            winner = rand.masked_fill(~in_cell, float("-inf")).argmax(-1)  # [B]
            has = in_cell.any(-1)                        # [B] 该样本在此格是否有点
            win_priority = 2.0 - seq / n_cells           # ∈ (1, 2]
            old = priority.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
            priority.scatter_(
                -1, winner.unsqueeze(-1),
                torch.where(has, torch.full_like(old, win_priority), old).unsqueeze(-1))
        return priority.topk(S, dim=-1)[1]               # [B, S]

    def forward(self, xytp, points):
        B, N, C = xytp.shape
        S = self.groups
        # 修复B：选点网格必须用 fps_parameter 缩放前的原始 x,y（∈[0,1]）。
        # fps_parameter 可学习，训练漂移后缩放坐标越界会被 clamp 静默压进
        # 边缘格；缩放坐标仍用于下游（new_xytp/分组坐标），只改选点来源
        sel_xytp = xytp
        xytp = self.fps_parameter * xytp
        xytp = xytp.contiguous()
        # ---- 唯一改动：FPS → x,y 空间均匀网格采样 ----
        fps_idx = self._spatial_uniform_idx(sel_xytp, self.groups).long()
        fps_idx, indices = torch.sort(fps_idx, dim=1)
        new_xytp = index_points(xytp, fps_idx)
        new_points = index_points(points, fps_idx)
        # ---- 以下与原 LocalGrouper 逐行一致 ----
        # ---- KNN：与官方全文对齐（Codex 2026-10-15 必改项，同 modules.py） ----
        # 两步距离公式不一致已修：全文 argsort 替代分块（显存算过 V100-16GB 放得下）。
        knn_k = self.kneighbors
        dists = square_distance(new_points, points)  # [B, S, N]
        idx = dists.argsort()[:, :, :knn_k]  # [B, S, K]
        idx = idx.sort(dim=-1)[0]
        grouped_xytp = index_points(xytp, idx)
        new_xytp = torch.mean(grouped_xytp, dim=2, keepdim=False)
        grouped_points = index_points(points, idx)
        grouped_points = torch.cat([grouped_points, grouped_xytp], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xytp], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xytp, new_points


class HyperbolicUniformGrouper(LocalGrouper):
    """双曲选点版 LocalGrouper（版本6 --hyperbolic_selector）。

    与 RP1SpatialUniformGrouper 同构，唯一区别是选点网格定义在 Poincaré
    盘的双曲度量上：欧氏坐标 (x,y)∈[0,1]^2 → u=(2x-1,2y-1)∈(-1,1)^2，
    双曲半径 r_h = 2·atanh(|u|)（近边界处双曲半径急剧增长 → 外圈格更密），
    角度 θ = atan2。网格在 (r_h, θ) 上等分 = 等双曲半径网格（外圈更密），
    体现"同样选点逻辑、不同度量空间"（学长：欧氏 vs 双曲特征不同）。

    其余（KNN、归一化、特征拼接）与原 LocalGrouper 逐行一致。

    use_scaled_coords=True（--hsel_scaled_coords，2026-10-14 ai2）：
    复现旧版 bug 行为——选点用 fps_parameter 缩放后的坐标。这是 2×2
    矩阵（网格几何 × 选点坐标口径）"双曲网格+缩放坐标"象限的受控
    复现开关：seed42 时该象限 +4.71pp（p=0.0002）为最强组件，需要
    跨种子复核。默认 False（修复版：缩放前原始坐标选点）。
    """

    def __init__(self, *args, use_scaled_coords: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_scaled_coords = use_scaled_coords

    @staticmethod
    def _hyperbolic_uniform_idx(xytp: torch.Tensor, S: int) -> torch.Tensor:
        """Poincaré 盘等双曲半径网格采样 S 个点索引（外圈更密）。

        Args:
            xytp: [B, N, 4]（t,x,y,p），x,y 已归一化到 [0,1]
            S: 需要的 center 数（= group_number[i]）
        Returns:
            [B, S] 选中点索引（未排序，调用方排序）
        """
        B, N, _ = xytp.shape
        N_r = int(math.ceil(math.sqrt(S)))
        N_theta = int(math.ceil(S / N_r))
        n_cells = N_r * N_theta
        eps = 1e-6
        # 欧氏盘坐标 u ∈ (-1,1)
        ux = xytp[..., 1].clamp(0.0, 1.0 - eps) * 2.0 - 1.0   # [B,N]
        uy = xytp[..., 2].clamp(0.0, 1.0 - eps) * 2.0 - 1.0   # [B,N]
        r_euc = torch.sqrt(ux * ux + uy * uy).clamp(max=1.0 - eps)
        r_h = 2.0 * torch.atanh(r_euc)                        # 双曲半径 [0, +inf)
        r_max = r_h.max(dim=-1, keepdim=True).values.clamp(min=eps)  # [B,1] per-sample
        r_cell = ((r_h / r_max) * N_r).long().clamp(max=N_r - 1)     # [B,N]
        theta = torch.atan2(uy, ux)                            # [-pi, pi]
        theta_cell = ((theta + math.pi) / (2 * math.pi) * N_theta).long().clamp(max=N_theta - 1)
        cell = r_cell * N_theta + theta_cell                   # [B,N]
        rand = torch.rand(B, N, device=xytp.device)
        priority = rand.clone()
        cell_order = torch.randperm(n_cells, device=xytp.device)
        for seq, j in enumerate(cell_order.tolist()):
            in_cell = (cell == j)
            if not in_cell.any():
                continue
            winner = rand.masked_fill(~in_cell, float("-inf")).argmax(-1)  # [B]
            has = in_cell.any(-1)
            win_priority = 2.0 - seq / n_cells                 # ∈ (1, 2]
            old = priority.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
            priority.scatter_(
                -1, winner.unsqueeze(-1),
                torch.where(has, torch.full_like(old, win_priority), old).unsqueeze(-1))
        return priority.topk(S, dim=-1)[1]                     # [B, S]

    def forward(self, xytp, points):
        B, N, C = xytp.shape
        S = self.groups
        # 修复B：选点网格用缩放前原始 x,y（∈[0,1]）；缩放坐标仍用于下游。
        # --hsel_scaled_coords=True 时改用缩放后坐标选点（复现旧版 bug
        # 行为，2×2 矩阵"双曲+缩放"象限的受控复现，见类 docstring）
        xytp_original = xytp
        xytp = self.fps_parameter * xytp
        sel_xytp = xytp if self.use_scaled_coords else xytp_original
        xytp = xytp.contiguous()
        # ---- 唯一改动：FPS → Poincaré盘等双曲半径网格采样 ----
        fps_idx = self._hyperbolic_uniform_idx(sel_xytp, self.groups).long()
        fps_idx, indices = torch.sort(fps_idx, dim=1)
        new_xytp = index_points(xytp, fps_idx)
        new_points = index_points(points, fps_idx)
        # ---- 以下与原 LocalGrouper 逐行一致 ----
        # ---- KNN：与官方全文对齐（Codex 2026-10-15 必改项，同 modules.py） ----
        # 两步距离公式不一致已修：全文 argsort 替代分块（显存算过 V100-16GB 放得下）。
        knn_k = self.kneighbors
        dists = square_distance(new_points, points)  # [B, S, N]
        idx = dists.argsort()[:, :, :knn_k]  # [B, S, K]
        idx = idx.sort(dim=-1)[0]
        grouped_xytp = index_points(xytp, idx)
        new_xytp = torch.mean(grouped_xytp, dim=2, keepdim=False)
        grouped_points = index_points(points, idx)
        grouped_points = torch.cat([grouped_points, grouped_xytp], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xytp], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xytp, new_points


class HyperbolicFusionBranch(nn.Module):
    """傅里叶+双曲并行融合分支（版本5 --hyperbolic_fusion）。

    与前 4 个组件不同，本组件不动傅里叶聚合路径，而是给每个 stage 新增
    一条并行的轻量双曲聚合支路，输出与傅里叶聚合输出 F_fourier 做残差
    门控融合：

        F_out = F_fourier + sigmoid(Linear(F_fourier)) * F_hyper

    双曲聚合（Lorentz 模型，核心流程复制自 local_remote_grouper.
    HyperbolicRemoteAggregator 的 lorentz 分支，按轻量化要求加 64 维瓶颈）：

      瓶颈降维 → exp_o 到 Lorentz 流形 → 组内质心锚定的双曲距离注意力 →
      切空间 Möbius 线性变换 → 注意力加权聚合 → 投影回 C 维

    输入是 stage 的 local 分组特征（LocalGrouper 输出，含傅里叶路径要用的
    全部信息，且在傅里叶支路之前取出，不影响其任何计算）。

    轻量化：hidden_dim=64 瓶颈使三 stage 合计约 13.5 万参数（< 总量 10%）。
    门控 weight 零初始化 + bias=-5，训练初期 gate≈sigmoid(-5)≈0.007，
    双曲支路近似关闭、不扰动原始傅里叶性能；稳定后随梯度逐渐打开，
    双曲无效时 gate 可自行收敛回 0（残差门控，不掉点）。

    曲率可学习（abs+1e-4 保证为正，初始 0.4）；切空间径向约束惩罚缓存在
    last_radial_pen，供训练循环加正则（同 HyperbolicRemoteAggregator）。
    """

    def __init__(self, channels: int, hidden_dim: int = 64, curvature_init: float = 0.4,
                 temperature: float = 1.0, gate_bias: float = -5.0):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.curvature = nn.Parameter(torch.tensor(float(curvature_init)))
        self.temperature = temperature
        # 瓶颈降维：C → d_h（双曲空间内的全部计算都在 d_h 维）
        self.in_proj = nn.Linear(channels, hidden_dim)
        # Möbius 线性变换：切空间 W·log_o(x) + b 再 exp 回双曲（等价 W ⊗_c x ⊕_c b）
        self.mobius_linear = nn.Linear(hidden_dim, hidden_dim)
        self.mobius_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, channels)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # 残差门控：gate = sigmoid(Linear(F_fourier))，逐点标量门（广播到 C 维）
        self.gate_proj = nn.Linear(channels, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))
        # Lorentz 徻向约束正则（每次 forward 后更新，供训练循环加到 loss）
        self.radial_radius = 4.0  # R_t
        self.last_radial_pen = None

    def _curvature(self) -> torch.Tensor:
        """曲率需为正，用 abs+eps 保证（curvature_init=0.4 时实际曲率≈0.4）。"""
        return torch.abs(self.curvature) + 1e-4

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """双曲聚合（单个 batch 的全部组一次算完）。

        Args:
            x: [B, S, K, C]，每个 FPS 中心的 K 个 local 邻居特征
        Returns:
            [B, S, C]，双曲质心聚合后的特征
        """
        B, S, K, C = x.shape
        c = self._curvature()
        # 瓶颈降维后进入 Lorentz 流形
        h = self.in_proj(x.reshape(-1, K, C))              # [M=B*S, K, d_h]
        h_hyp = lorentz_exp_origin(h, c)                   # [M, K, d_h+1]
        # 质心锚定的双曲距离注意力（离质心越近权重越大）
        centroid = h.mean(dim=1, keepdim=True)             # [M, 1, d_h]
        anchor = lorentz_exp_origin(centroid, c)           # [M, 1, d_h+1]
        distances = lorentz_distance(anchor.expand_as(h_hyp), h_hyp, c)  # [M, K]
        attn_weights = F.softmax(-distances / self.temperature, dim=-1)
        # 切空间 Möbius 线性变换
        h_tangent = lorentz_log_origin(h_hyp, c)           # [M, K, d_h]
        h_transformed = self.mobius_linear(h_tangent) + self.mobius_bias
        h_hyp_transformed = lorentz_exp_origin(h_transformed, c)
        # 注意力加权聚合（切空间）后映射回欧氏
        agg_tangent = torch.bmm(
            attn_weights.unsqueeze(1), lorentz_log_origin(h_hyp_transformed, c)
        ).squeeze(1)                                       # [M, d_h]
        agg_hyp = lorentz_exp_origin(agg_tangent, c)
        agg_euclidean = lorentz_log_origin(agg_hyp, c)     # [M, d_h]
        # Lorentz 径向约束：切向量范数不超过 R_t，超出部分二次惩罚（hinge^2）
        R_t = self.radial_radius
        pen_x = F.relu(h_tangent.norm(dim=-1) - R_t) ** 2  # [M, K]
        pen_agg = F.relu(agg_tangent.norm(dim=-1) - R_t) ** 2  # [M]
        self.last_radial_pen = (pen_x.mean() + pen_agg.mean()) / 2
        out = self.out_proj(self.layer_norm(agg_euclidean))  # [M, C]
        return out.reshape(B, S, C)

    def forward(self, x_local: torch.Tensor, f_fourier: torch.Tensor) -> torch.Tensor:
        """残差门控融合。

        Args:
            x_local: [B, S, K, C] stage 的 local 分组特征（双曲支路输入）
            f_fourier: [B, S, C] 傅里叶聚合（gated fusion）输出 F_fourier
        Returns:
            [B, S, C] F_out = F_fourier + gate * F_hyper
        """
        f_hyper = self.aggregate(x_local)                  # [B, S, C]
        gate = torch.sigmoid(self.gate_proj(f_fourier))    # [B, S, 1]，初始 ≈ 0
        return f_fourier + gate * f_hyper



def mobius_add(x, y, c, eps=1e-5):
    """Poincaré 球上的 Möbius 向量加法：x ⊕_c y（x, y 均为球内点）。

    (1 + 2c⟨x,y⟩ + c‖y‖²)·x + (1 - c‖x‖²)·y
    ─────────────────────────────────────────────
      1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²
    """
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return clip_to_disk(num / denom.clamp_min(eps), c)


class HyperbolicEmbedding(nn.Module):
    """双曲嵌入瓶颈层：欧氏→双曲→双曲空间线性变换+Möbius偏置→欧氏，残差连接

    原实现 exp_map(log_map(x)) 首尾配对恒等，双曲部分无效；现改为
    进双曲一次 exp_map、出双曲一次 log_map，中间全程在 Poincaré 球内运算：
      p  = exp_o(x)                          欧氏 → 双曲
      p' = exp_o(W · log_o(p))               双曲线性（切空间线性后回球）
      p'' = p' ⊕_c b                         双曲空间 Möbius 加可学习偏置
      out = log_o(p'')                       双曲 → 欧氏
    """
    def __init__(self, dim, curvature_init=0.4):
        super(HyperbolicEmbedding, self).__init__()
        # 修复C：期望曲率语义——参数初始化为 softplus⁻¹(curvature_init)，
        # get_c 的 softplus(参数) 才真正得到初始曲率 0.4
        # （原先直接存 0.4，softplus(0.4)≈0.913 ≠ 0.4，与注释不符）
        c_init = math.log(math.expm1(curvature_init))
        self.curvature = nn.Parameter(torch.tensor(c_init))
        self.linear = nn.Linear(dim, dim)
        # Möbius 偏置：球内点，零初始化（原点 ⊕ p = p，初始为恒等）
        self.bias = nn.Parameter(torch.zeros(dim))
        self.layer_norm = nn.LayerNorm(dim)

    def get_c(self):
        return F.softplus(self.curvature) + 1e-4

    def forward(self, x):
        c = self.get_c()
        # 修复E：进球前尺度归一——exp_o 输出范数 = tanh(√c·‖x‖/2)/√c，
        # ‖x‖ 大时贴着球边界饱和（tanh 梯度 ~1e-6，曲率学不动）。
        # 归一到 √dim 的温和尺度（只改幅度不改方向），保证 tanh 工作在
        # 非饱和区，曲率/线性层梯度正常回传
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6) * x.shape[-1] ** 0.5
        p = poincare_exp_origin(x_norm, c)               # 进双曲
        p = poincare_exp_origin(self.linear(poincare_log_origin(p, c)), c)  # 双曲线性
        p = mobius_add(p, clip_to_disk(self.bias, c), c)  # 双曲偏置
        x_out = poincare_log_origin(p, c)                # 出双曲
        return self.layer_norm(x + x_out)



class HyperbolicAttention(nn.Module):
    """双曲自注意力残差支路：用Lorentz双曲距离计算注意力权重。

    在SECNet的傅里叶聚合（gated fusion 输出 F_fourier）之后，额外加一条
    双曲自注意力残差支路（x + 0.1 * x_hyper），不替换原有欧氏聚合路径，
    原始计算逐字节不变，双曲支路以小权重残差注入，利用双曲空间的层级结构特性。
    """
    def __init__(self, channels, curvature_init=0.4):
        super(HyperbolicAttention, self).__init__()
        # 修复C：同 HyperbolicEmbedding——期望曲率语义，softplus(参数)=curvature_init
        c_init = math.log(math.expm1(curvature_init))
        self.curvature = nn.Parameter(torch.tensor(c_init))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def _curvature(self):
        return F.softplus(self.curvature) + 1e-4
    
    def lorentz_distance(self, x, y):
        """计算Lorentz双曲距离。
        x, y: (B, N, C) 欧氏特征
        返回: (B, N, N) 双曲距离矩阵
        """
        c = self._curvature()
        # 欧氏到Lorentz映射: (x0, x_vec) where x0 = sqrt(1/c + ||x_vec||^2)
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # (B, N, 1)
        x0 = torch.sqrt(1.0 / c + x_norm ** 2)  # (B, N, 1)
        x_lorentz = torch.cat([x0, x], dim=-1)  # (B, N, C+1)
        
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        y0 = torch.sqrt(1.0 / c + y_norm ** 2)
        y_lorentz = torch.cat([y0, y], dim=-1)
        
        # Lorentz内积: <x,y>_L = -x0*y0 + x_vec · y_vec
        inner = -x_lorentz[..., 0:1] * y_lorentz[..., 0:1].transpose(-2, -1) +                 torch.matmul(x_lorentz[..., 1:], y_lorentz[..., 1:].transpose(-2, -1))
        
        # Lorentz距离: d_L = (1/√c)·arccosh(-c·⟨x,y⟩_L)，与 local_remote_grouper.lorentz_distance 对齐
        inner_clamped = torch.clamp(-c * inner, min=1.0 + 1e-6)
        dist = torch.acosh(inner_clamped) / torch.sqrt(c)
        return dist
    
    def forward(self, query, key, value):
        """
        query: (B, N, C)
        key: (B, N, C)
        value: (B, N, C)
        返回: (B, N, C) 加权聚合后的特征
        """
        # 双曲距离矩阵
        dist = self.lorentz_distance(query, key)  # (B, N, N)
        
        # 注意力权重: softmax(-dist / temperature)
        attn = F.softmax(-dist / (self.temperature.abs() + 1e-6), dim=-1)
        
        # 加权聚合
        out = torch.matmul(attn, value)  # (B, N, C)
        return out



class HyperbolicMLP(nn.Module):
    """双曲 MLP（P0-2 --hyperbolic_mlp）：Möbius 线性 + 偏置 + 双曲激活。

    动机：HyperbolicAttention 的输出（欧氏 value 聚合结果）直接以欧氏残差
    注入，双曲几何的优势可能被后续欧氏变换破坏。本模块让特征"真在双曲里
    做变换"：插在注意力输出之后、残差注入之前，全程 Poincaré 球内运算
    （复用 poincare_exp_origin / poincare_log_origin / mobius_add）：

      p    = exp_o(norm(x))                 欧氏 → 双曲
      p'   = exp_o(W · log_o(p))            Möbius 矩阵乘 W ⊗_c p
      p''  = p' ⊕_c b                       Möbius 偏置加
      p''' = exp_o(ReLU(log_o(p'')))        双曲激活
      out  = log_o(p''')                    双曲 → 欧氏

    参数量控制（P0-2 预算 <100K）：W 用低秩分解 W = U·Vᵀ（rank=32），
    全秩 C×C 三 stage 合计 132²+268²+540²≈38 万会超预算，低秩后三 stage
    合计 65*(132+268+540)+3 = 61103 个参数。
    W 的 U 半边与 b 均零初始化 → W=0、b=原点，初始输出恒为 0，
    残差注入后与 --hyperbolic_attention 单开的初始数值完全一致，
    训练初期不扰动原始性能（V 用小随机初始化保证 U 有非零梯度）。
    """

    def __init__(self, dim, rank=32, curvature_init=0.4):
        super(HyperbolicMLP, self).__init__()
        # 修复C 同款语义：softplus(参数) = 初始曲率 0.4
        c_init = math.log(math.expm1(curvature_init))
        self.curvature = nn.Parameter(torch.tensor(c_init))
        # Möbius 矩阵乘低秩分解：log_o(p) @ U @ Vᵀ（U 零初始化 → W=0）
        self.weight_u = nn.Parameter(torch.zeros(dim, rank))
        self.weight_v = nn.Parameter(torch.randn(dim, rank) / math.sqrt(rank))
        # Möbius 偏置：球内点，零初始化（原点 ⊕ p = p，初始恒等）
        self.bias = nn.Parameter(torch.zeros(dim))

    def get_c(self):
        return F.softplus(self.curvature) + 1e-4

    def forward(self, x):
        """
        x: (B, S, C) 欧氏特征（HyperbolicAttention 输出）
        返回: (B, S, C) 双曲变换后回欧氏的特征
        """
        c = self.get_c()
        # 进球前尺度归一（同 HyperbolicEmbedding 修复E，防 tanh 饱和）
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6) * x.shape[-1] ** 0.5
        p = poincare_exp_origin(x_norm, c)               # 欧氏 → 双曲
        h = poincare_log_origin(p, c) @ self.weight_u    # [.., rank]
        p = poincare_exp_origin(h @ self.weight_v.t(), c)  # Möbius 矩阵乘，回球
        p = mobius_add(p, clip_to_disk(self.bias, c), c)   # Möbius 偏置加
        p = poincare_exp_origin(F.relu(poincare_log_origin(p, c)), c)  # 双曲激活
        return poincare_log_origin(p, c)                 # 双曲 → 欧氏



class MambaSSM(nn.Module):
    """块内并行+块间串行版Mamba（数值稳定，训练快）。

    h_s = dA_s * h_{s-1} + dBx_s 的线性递推：
    - 块内（chunk=32）用对数cumsum并行求解，无串行循环
    - 块间（n_chunks次）串行传播状态
    关键约束：dt.clamp(max=0.25) + A=-1..-4 保证块内 exp(-cum_log) <= exp(32) 不溢出。
    循环次数 = n_chunks（S=512时16次），远快于全串行。
    """
    def __init__(self, d_model, d_state=4, chunk=32):
        super(MambaSSM, self).__init__()
        hidden = min(d_model, 64)
        self.chunk = chunk
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.conv1d = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, groups=hidden)
        self.x_proj = nn.Linear(hidden, d_state)
        self.dt_proj = nn.Linear(d_state, hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        A = torch.arange(1, d_state + 1, dtype=torch.float)
        self.register_buffer('A', -A)
        self.D = nn.Parameter(torch.ones(hidden))

    def forward(self, x):
        B, S, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
        x = F.silu(x)
        B_proj = self.x_proj(x)
        dt = F.softplus(self.dt_proj(B_proj)).clamp(max=0.25)
        dA = torch.exp(dt.unsqueeze(-1) * self.A.view(1, 1, 1, -1))
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)
        dBx = dB * x.unsqueeze(-1)

        # 块内并行（对数cumsum）
        log_dA = torch.log(dA + 1e-8)
        n_chunks = (S + self.chunk - 1) // self.chunk
        pad = n_chunks * self.chunk - S
        if pad > 0:
            log_dA_pad = F.pad(log_dA, (0, 0, 0, 0, 0, pad))
            dBx_pad = F.pad(dBx, (0, 0, 0, 0, 0, pad))
        else:
            log_dA_pad, dBx_pad = log_dA, dBx
        log_dA_c = log_dA_pad.view(B, n_chunks, self.chunk, -1, self.A.shape[0])
        dBx_c = dBx_pad.view(B, n_chunks, self.chunk, -1, self.A.shape[0])

        cum_log = torch.cumsum(log_dA_c, dim=2)  # 块内前缀log转移
        pre = torch.exp(-cum_log) * dBx_c
        pre_cum = torch.cumsum(pre, dim=2)
        local = torch.exp(cum_log) * pre_cum  # 块内从0状态的前缀状态
        A_pref = torch.exp(cum_log)  # 块内前缀转移

        # 块间串行传播（n_chunks次）
        h = torch.zeros(B, x.shape[-1], self.A.shape[0], device=x.device, dtype=x.dtype)
        outs = []
        for c in range(n_chunks):
            global_c = h.unsqueeze(1) * A_pref[:, c] + local[:, c]
            outs.append(global_c)
            h = global_c[:, -1]
        h_full = torch.cat(outs, dim=1)[:, :S]  # (B, S, hidden, d_state)

        y = (h_full * self.A.view(1, 1, 1, -1)).sum(-1) + self.D * x
        y = y * F.silu(z)
        return self.out_proj(y)

class FECNet(nn.Module):
    def __init__(self, num_classes, feature_list=None, group_number=None,
                 use_group_norm: bool = False,
                 use_time_enc: bool = False,
                 use_kda: bool = False,
                 use_rp1_selector: bool = False,
                 use_hyperbolic_selector: bool = False,
                 use_hyperbolic_classifier: bool = False,
                 use_hyperbolic_fusion: bool = False,
                 use_hyperbolic_embedding: bool = False,
                 use_hyperbolic_attention: bool = False,
                 use_hyperbolic_mlp: bool = False,
                 use_event_mamba_ssm: bool = False,
                 use_spike_time_enc: bool = False,
                 use_scale_norm_fps: bool = False,
                 use_hsel_scaled_coords: bool = False):
        super().__init__()
        _enc_flags = [use_time_enc, use_kda, use_spike_time_enc]
        if sum(_enc_flags) > 1:
            raise ValueError("time_enc / kda / spike_time_enc 互斥，只能选一个")
        # 修复A：两个 selector 同开时下方 if/elif 会静默走 rp1，必须显式报错
        if use_rp1_selector and use_hyperbolic_selector:
            raise ValueError("rp1_selector 与 hyperbolic_selector 互斥，只能选一个")
        # 2026-10-14 ai2：--hsel_scaled_coords 只对 --hyperbolic_selector 有意义
        # （复现旧版 bug：双曲网格选点用 fps_parameter 缩放后坐标）。
        # 不带 hyperbolic_selector 时按 no-op 并警告，不抛错（推理工具
        # 重建模型时统一传所有 flag，避免旧 log 缺键重建失败）。
        if use_hsel_scaled_coords and not use_hyperbolic_selector:
            print('[FECNet] 警告: use_hsel_scaled_coords 需与 use_hyperbolic_selector '
                  '同开才有意义，当前按 no-op 处理')
        self.use_hsel_scaled_coords = use_hsel_scaled_coords
        # 组件版本开关（消融实验，默认全关 = 原始 SECNet）
        self.use_time_enc = use_time_enc
        self.use_kda = use_kda
        # 版本5 --hyperbolic_fusion：傅里叶+双曲并行融合（残差门控），
        # 默认关闭时不创建任何双曲模块，state_dict 键与原始 SECNet 完全一致
        self.use_hyperbolic_fusion = use_hyperbolic_fusion
        self.use_hyperbolic_embedding = use_hyperbolic_embedding
        self.use_hyperbolic_attention = use_hyperbolic_attention
        # P0-2 --hyperbolic_mlp：双曲注意力输出后的双曲 MLP（Möbius 线性+偏置+
        # 双曲激活），默认关闭时不创建任何模块，state_dict 键与原始一致。
        # 仅与 --hyperbolic_attention 组合才有意义（要有双曲注意力输出可变换）
        self.use_hyperbolic_mlp = use_hyperbolic_mlp
        if use_hyperbolic_mlp and not use_hyperbolic_attention:
            print('[FECNet] 警告: use_hyperbolic_mlp 需与 use_hyperbolic_attention '
                  '同开才有意义（否则没有双曲注意力输出可变换），当前按 no-op 处理')
        self.use_event_mamba_ssm = use_event_mamba_ssm
        self.use_spike_time_enc = use_spike_time_enc
        if use_hyperbolic_fusion:
            self.hyper_fusion_list = nn.ModuleList()
        # 官方 N-Caltech101 配置（与 best_checkpoint 维度一致）
        self.feature_list = feature_list if feature_list is not None else [64, 132, 268, 540]
        self.group_number = group_number if group_number is not None else [2048, 1024, 512, 256]
        self.neighbors = [24, 24, 24, 24]
        self.stages = 3
        self.local_grouper_list = nn.ModuleList()
        self.fft_temporal_weight_list = nn.ParameterList()
        self.fft_sptaial_weight_list = nn.ParameterList()
        self.fft_temporal_list = nn.ModuleList()
        self.fft_sptaial_list = nn.ModuleList()
        # embed 输入维度：原始 4 (t,x,y,p)；time_enc 时 t → 8 维编码 = 11；
        # kda 时 t → 16 维编码 = 19
        if use_kda:
            self.kda_point_enc = KDAPointEncoder(kneighbors=24, state_dim=16)
            embed_in_dim = 19
        elif use_time_enc:
            embed_in_dim = 11
        elif use_spike_time_enc:
            embed_in_dim = 12  # xy+t+p+8 sin/cos
        else:
            embed_in_dim = 4
        self.embed_dim = Linear1Layer(embed_in_dim, self.feature_list[0], 1, use_group_norm=use_group_norm)
        self.group_conv_list = nn.ModuleList()
        self.aggregation_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.bn_local_list = nn.ModuleList()
        self.bn_global_list = nn.ModuleList()
        self.attention = Attention(self.feature_list[-1])
        for i in range(self.stages):
            if use_rp1_selector:
                local_grouper = RP1SpatialUniformGrouper(self.feature_list[i], self.group_number[i], 24, True, "center", use_scale_norm_fps=use_scale_norm_fps)
            elif use_hyperbolic_selector:
                local_grouper = HyperbolicUniformGrouper(self.feature_list[i], self.group_number[i], 24, True, "center", use_scale_norm_fps=use_scale_norm_fps, use_scaled_coords=use_hsel_scaled_coords)
            else:
                local_grouper = LocalGrouper(self.feature_list[i], self.group_number[i], 24, True, "center", use_scale_norm_fps=use_scale_norm_fps)
            self.local_grouper_list.append(local_grouper)
            aggregation = Attention(self.feature_list[i + 1])
            self.aggregation_list.append(aggregation)
            fft_sptaial_weight = nn.Parameter(torch.cat(
                (torch.ones(self.feature_list[i + 1] // 2 + 1, self.neighbors[i], 1, dtype=torch.float32),
                 torch.zeros(self.feature_list[i + 1] // 2 + 1, self.neighbors[i], 1, dtype=torch.float32)),
                dim=-1))
            self.fft_sptaial_list.append(fftlayer(fft_sptaial_weight))
            fft_temporal_weight = nn.Parameter(torch.cat(
                (torch.ones(self.group_number[i] // 2 + 1, self.feature_list[i + 1], 1, dtype=torch.float32),
                 torch.zeros(self.group_number[i] // 2 + 1, self.feature_list[i + 1], 1, dtype=torch.float32)),
                dim=-1))
            self.fft_temporal_list.append(fftlayer(fft_temporal_weight))
            conv = Linear2Layer(self.feature_list[i + 1], 1, 1, use_group_norm=use_group_norm)
            self.conv_list.append(conv)
            group_conv = Linear2Layer(self.feature_list[i + 1], 1, 1, use_group_norm=use_group_norm)
            self.group_conv_list.append(group_conv)
            self.bn_local_list.append(make_norm1d(self.feature_list[i + 1], use_group_norm))
            self.bn_global_list.append(make_norm1d(self.feature_list[i + 1], use_group_norm))
            if use_hyperbolic_fusion:
                # 版本5：并行双曲聚合支路（输入/输出均 feature_list[i+1] 维，
                # 64 维瓶颈轻量化，曲率初始 0.4，门控初始 ≈0）
                self.hyper_fusion_list.append(
                    HyperbolicFusionBranch(self.feature_list[i + 1],
                                           hidden_dim=64, curvature_init=0.4))
        if use_hyperbolic_classifier:
            # 版本4 --hyperbolic_classifier：双曲原型分类头（Lorentz 模型，
            # 复用 local_remote_grouper.HyperbolicPrototypeClassifier），
            # 测试时用双曲距离（logits = -lorentz_distance）分类
            self.classifier = HyperbolicPrototypeClassifier(
                input_dim=self.feature_list[-1] * 2,
                num_classes=num_classes,
                embed_dim=64,
                curvature_init=0.4)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.feature_list[-1] * 2, 256),
                make_norm1d(256, use_group_norm),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        if self.use_hyperbolic_embedding:
            self.hyper_embed = HyperbolicEmbedding(self.feature_list[-1] * 2)
        if self.use_event_mamba_ssm:
            # EventMamba风格SSM：只在stage3（最后一级）做状态空间建模（S=512，控制开销），
            # 只建最后一级的 SSM，前几级不建（旧版建3个但只用 stage3，浪费参数）
            self.mamba_ssms = nn.ModuleList([
                MambaSSM(self.feature_list[self.stages])
            ])
        if self.use_hyperbolic_attention:
            # 双曲注意力残差支路：每个stage傅里叶聚合后额外加一条
            # Lorentz 双曲距离自注意力残差（不替换原欧氏聚合路径）
            self.hyper_attns = nn.ModuleList([
                HyperbolicAttention(self.feature_list[i + 1])
                for i in range(self.stages)
            ])
            if self.use_hyperbolic_mlp:
                # P0-2：每 stage 一个双曲 MLP（Möbius 线性 rank=32 低秩分解，
                # W/b 零初始化，初始输出恒为 0），串在双曲注意力输出之后
                self.hyper_mlps = nn.ModuleList([
                    HyperbolicMLP(self.feature_list[i + 1])
                    for i in range(self.stages)
                ])

    def forward(self, x: torch.Tensor):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # 组件版本的 embed 输入变换（xyz 保持 [B,N,4] 原样传给 grouper，
        # FPS/KNN/傅里叶聚合的坐标语义不变；默认全关时 embed_in = x 原样）
        embed_in = x
        if self.use_time_enc:
            # 版本1 --time_enc：t → 8 维 sin/cos 多频编码（频率 2^0..2^3），
            # 输入 (x,y,p + 8) = 11 维
            t_enc = time_freq_encode(xyz, num_freqs=4)          # [B, N, 8]
            embed_in = torch.cat([xyz[..., 1:4], t_enc], dim=-1).permute(0, 2, 1)  # [B, 11, N]
        elif self.use_spike_time_enc:
            # 版本7 --spike_time_enc：SpikePoint 混合时间编码，
            # 保留原始 t 作坐标 + 额外 8 维 sin/cos 通道，输入 (x,y,t,p + 8) = 12 维
            t_enc = time_freq_encode(xyz, num_freqs=4)          # [B, N, 8]
            embed_in = torch.cat([xyz, t_enc], dim=-1).permute(0, 2, 1)  # [B, 12, N]
        elif self.use_kda:
            # 版本2 --kda：t → 16 维 KDA 因果时间编码，
            # 输入 (x,y,p + 16) = 19 维
            kda_feat = self.kda_point_enc(xyz)                  # [B, N, 16]
            embed_in = torch.cat([xyz[..., 1:4], kda_feat], dim=-1).permute(0, 2, 1)  # [B, 19, N]
        x = self.embed_dim(embed_in)
        x = x.permute(0, 2, 1)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x)
            # 版本5 --hyperbolic_fusion：local 分组特征先送入双曲支路（副本引用，
            # 不修改 x 本身，傅里叶支路随后的所有计算逐字节不变）
            if self.use_hyperbolic_fusion:
                f_hyper = self.hyper_fusion_list[i].aggregate(x)  # [B, S, C]
            x = x.permute(0, 1, 3, 2)
            b, n, d, s = x.size()
            x = x.reshape(-1, d, s)
            x = self.fft_sptaial_list[i](x)
            x = F.gelu(x)
            x = x.permute(0, 2, 1)
            att = self.aggregation_list[i](x)
            x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
            x = x.reshape(b, n, -1)
            # 双曲注意力残差支路：傅里叶聚合后额外加 Lorentz 双曲距离自注意力，
            # 不替换原路径，小权重残差，训练初期不影响原始性能
            if self.use_hyperbolic_attention:
                x_hyper = self.hyper_attns[i](x, x, x)  # (B, S, C)
                if self.use_hyperbolic_mlp:
                    # P0-2：注意力输出先进 Poincaré 球做 Möbius 线性+偏置+
                    # 双曲激活再回欧氏（W/b 零初始化，初始输出恒为 0，
                    # 初始数值与 --hyperbolic_attention 单开完全一致），
                    # 之后仍是 0.1 小权重残差注入
                    x_hyper = self.hyper_mlps[i](x_hyper)
                x = x + 0.1 * x_hyper  # 小权重残差，训练初期不影响原始性能
            x = self.fft_temporal_list[i](x)
            x = x.permute(0, 2, 1)
            x = F.gelu(x)
            # local branch
            x_local = self.conv_list[i](x)
            x_local = self.bn_local_list[i](x_local)
            # global branch: mean pool -> conv -> expand -> gate
            x_global = F.adaptive_avg_pool1d(x, 1)
            x_global = self.group_conv_list[i](x_global)
            x_global = self.bn_global_list[i](x_global)
            x_global = x_global.expand(-1, -1, n)
            # gated fusion
            x = x_local * torch.sigmoid(x_global)
            x = x.permute(0, 2, 1)  # [B, S, C] = F_fourier
            if self.use_event_mamba_ssm and i == self.stages - 1:
                x_ssm = self.mamba_ssms[0](x)
                x = x + 0.1 * x_ssm
            # 版本5 --hyperbolic_fusion：残差门控融合
            # F_out = F_fourier + sigmoid(Linear(F_fourier)) * F_hyper（gate 初始 ≈0）
            if self.use_hyperbolic_fusion:
                gate = torch.sigmoid(self.hyper_fusion_list[i].gate_proj(x))  # [B, S, 1]
                x = x + gate * f_hyper
        x_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1)
        x_max = x_max.reshape(batch_size, -1)
        x = x.mean(dim=1)
        x = torch.cat([x, x_max], dim=1)
        if self.use_hyperbolic_embedding:
            x = self.hyper_embed(x)
        x = self.classifier(x)
        return x


def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """Count flops for the rfft/rfftn operator."""
    input_shape = inputs[0].type().sizes()
    print(input_shape)
    B, N, C = input_shape
    flops = N * C * np.ceil(np.log2(N)) + N * C * 2
    flops = flops * B
    return flops


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    model = FECNet(num_classes=10)
    model.eval()
    input = torch.randn(1, 4, 1024)
    fca1 = FlopCountAnalysis(model, input)
    handlers = {
        'aten::fft_rfft': rfft_flop_jit,
        'aten::fft_irfft': rfft_flop_jit,
    }
    fca1.set_op_handle(**handlers)
    flops1 = fca1.total()
    print(fca1.by_module())
    print(flop_count_table(fca1, max_depth=1))
    print("#### GFLOPs: {}".format(flops1 / 1e9))
