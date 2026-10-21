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
    # G16 六处理用 v2（带 g16_treatment 分派 + P0-1 修复），旧版无六处理
    from local_remote_grouper_v2 import LocalRemoteGrouper as LocalRemoteGrouperV2
except ImportError:  # pragma: no cover
    from models.local_remote_grouper import (KDACausalEncoder, HyperbolicPrototypeClassifier,
                                             lorentz_exp_origin, lorentz_log_origin,
                                             lorentz_distance,
                                             poincare_exp_origin, poincare_log_origin,
                                             clip_to_disk)
    from models.local_remote_grouper_v2 import LocalRemoteGrouper as LocalRemoteGrouperV2


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


class HyperbolicGate(nn.Module):
    """双曲距离门控（替换式，替换 SECNet gated fusion 的通道门控）。

    原 gated fusion（每个 stage 末尾）：
        x_global = avg_pool(x) -> group_conv -> bn -> expand
        x = x_local * sigmoid(x_global)   # 逐通道门控，欧氏全局池化+线性学出

    本组件把门控信号的来源换成双曲距离：每个空间位置（组）的 C 维特征投影到
    Lorentz 流形，计算它离全体组质心的双曲距离，距离近（典型特征）门控大、
    距离远（离群特征）门控小：

        gate_s = sigmoid(alpha * (r0 - d_hyp(centroid, group_s)))   # [B, S]
        x      = x_local * gate                                      # 逐组门控

    语义：门控粒度从原「逐通道」（SE 式标量门控）改为「逐组/逐空间位置」。
    双曲距离是点到点的几何量，对应空间/层级结构（双曲几何的用武之地），
    而非通道重要性；逐通道标量无法承载双曲几何。保留「local × gate」骨架，
    只换门控信号来源（替换共识③：融合层 gated→双曲加权）。

    LayerNorm 稳定特征范数；alpha 用 softplus 参数化保证恒正（abs 在 0 点梯度
    为 0 会卡死 alpha），alpha_raw 初始 log(expm1(0.01)) -> alpha≈0.01、初始
    gate≈sigmoid(-0.01*dist)≈0.48 中性（对齐原 gated fusion 随机 sigmoid 均值），
    alpha/bias/曲率/in_proj 均有非零梯度。曲率 softplus 参数化初始 0.4（同
    HyperbolicAttention 修复C）。
    """

    def __init__(self, channels: int, hidden_dim: int = 64, curvature_init: float = 0.4):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        # 曲率：期望语义 softplus(param)=curvature_init（同 HyperbolicAttention 修复C）
        c_init = math.log(math.expm1(curvature_init))
        self.curvature = nn.Parameter(torch.tensor(c_init))
        # 瓶颈降维 C -> d_h（双曲空间内的全部计算在 d_h 维）
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # 门控 logit = bias - alpha*dist。alpha 用 softplus 参数化保证恒正
        # （abs() 在 0 点梯度为 0 会卡死 alpha 及整个双曲分支，见 Codex 复审 P0）：
        # alpha_raw 初始 log(expm1(0.01)) -> alpha=softplus(alpha_raw)=0.01
        # （小正、梯度非零，初始 gate≈sigmoid(-0.01*dist)≈0.48 中性）
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.01))))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _curvature(self) -> torch.Tensor:
        return F.softplus(self.curvature) + 1e-4

    def forward(self, x_local: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
        """双曲距离门控。

        Args:
            x_local: [B, C, S] local 分支输出（被门控的对象）
            x_src:   [B, C, S] 门控特征源（傅里叶时间聚合 + gelu 后、conv 前）
        Returns:
            [B, C, S] x_local * gate，gate 为 [B, 1, S] 逐组门控
        """
        B, C, S = x_src.shape
        c = self._curvature()
        # 每个空间位置 s 的 C 维特征 -> d_h 维瓶颈 -> Lorentz 流形
        h = self.in_proj(x_src.permute(0, 2, 1))        # [B, S, d_h]
        h = self.layer_norm(h)
        h_hyp = lorentz_exp_origin(h, c)                # [B, S, d_h+1]
        # 全体组的 Lorentz 质心（切空间均值再 exp，同 HyperbolicFusionBranch）
        centroid_tan = h.mean(dim=1, keepdim=True)      # [B, 1, d_h]
        centroid = lorentz_exp_origin(centroid_tan, c)  # [B, 1, d_h+1]
        dist = lorentz_distance(centroid.expand_as(h_hyp), h_hyp, c)  # [B, S]
        # 离质心越近门控越大（alpha>0，softplus 保证恒正、梯度非零）
        alpha = F.softplus(self.alpha_raw)
        gate = torch.sigmoid(self.bias - alpha * dist)  # [B, S]
        gate = gate.unsqueeze(1)                        # [B, 1, S] 广播到通道维
        return x_local * gate


class FcaGate(nn.Module):
    """频域通道门控（FcaNet 思路，替换 SECNet gated fusion 的通道门控）。

    原 gated fusion：x_global = avg_pool(x)（= 最低频 DCT 分量，即 DC/均值）
    -> group_conv -> sigmoid -> 逐通道门控。SECNet 门控本质是 SENet 的 SE 通道
    注意力，其「全局平均池化」只取了 DCT 的 DC 分量。

    本组件按 FcaNet（Frequency Channel Attention Networks, Qin et al. ICCV 2021）
    的思想，把「仅 DC 分量」扩展为「前 num_freq 个低频 DCT 分量」：每个通道用
    多个频率分量压缩成一个 num_freq 维描述子，再经 FC 学出逐通道门控。
    num_freq=1 时信息上等价于全局平均池化（DC 投影与 mean 成正比），但结构
    是 Linear-ReLU-Linear 而非原版 group_conv-bn，非严格退化。保留「逐通道
    门控」语义，不换粒度（与 HyperbolicGate 的逐组门控形成对照）。

    机制：
        freq_c = DCT-II(x_src[:, c, :])[:num_freq]       # [B, C, num_freq]
        gate   = sigmoid(FC(freq.reshape(B, C*num_freq))) # [B, C]
        x      = x_local * gate.unsqueeze(-1)            # 逐通道门控

    预计算 DCT-II 基矩阵（seq_len = group_number[i]，每 stage 不同），基向量
    L2 归一化使各频率分量尺度一致（否则 DC 分量量级是高频分量的 √S 倍）。
    bottleneck 降维控制参数量（≈ +12%）。
    """

    def __init__(self, channels: int, seq_len: int, num_freq: int = 4,
                 reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.num_freq = num_freq
        # 预计算 DCT-II 基矩阵 [num_freq, seq_len]，L2 归一化
        n = torch.arange(seq_len, dtype=torch.float32)
        i = torch.arange(num_freq, dtype=torch.float32)
        basis = torch.cos(math.pi * i.unsqueeze(1) * (n.unsqueeze(0) + 0.5) / seq_len)
        basis = F.normalize(basis, dim=1)                # 每行单位范数
        self.register_buffer('basis', basis)             # [num_freq, seq_len]
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels * num_freq, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x_local: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
        """频域通道门控。

        Args:
            x_local: [B, C, S] local 分支输出（被门控的对象）
            x_src:   [B, C, S] 门控特征源（傅里叶时间聚合 + gelu 后、conv 前）
        Returns:
            [B, C, S] x_local * gate，gate 为 [B, C, 1] 逐通道门控
        """
        B, C, S = x_src.shape
        if S != self.basis.shape[1]:
            raise ValueError("FcaGate seq_len mismatch: forward S=%d, basis built for %d"
                             % (S, self.basis.shape[1]))
        # 每个通道在 S 维上投影到 num_freq 个 DCT 频率分量
        basis = self.basis.to(x_src.dtype)               # 兼容 AMP（fp16）
        freq = torch.matmul(x_src, basis.t())            # [B, C, num_freq]
        freq = freq.reshape(B, C * self.num_freq)        # [B, C*num_freq]
        gate = torch.sigmoid(self.fc(freq))              # [B, C]
        return x_local * gate.unsqueeze(-1)              # [B, C, S]


class HyperbolicRadialRectify(nn.Module):
    """曲率感知双曲径向校正（CHR，PointCHR 思路，arXiv 2607.24052，2026-07）。

    PointCHR 核心：点云里高曲率/精细区域（物体边缘、复杂结构）在空间分布
    长尾稀疏，欧氏多项式体积增长使其被低曲率平滑区域淹没、难以分辨。利用
    双曲流形边界「指数体积增长」的特性，把复杂度高的点投影到边界（更大
    有效容量），缓解欧氏表示拥挤（representation crowding）。

    事件点云（t,x,y,p）无几何曲率，用可学习复杂度分数 κ 作代理：κ 大的点
    （离群/精细）获得更大径向半径（投影到边界），κ 小的点留在中心。与
    HyperbolicGate（离质心近→门控大）形成相反的设计：本组件「增强」离群点
    而非抑制。

    机制（Lorentz 流形内径向校正 + 残差；α 用 softplus 避免 abs 零点梯度坑）：
        κ    = sigmoid(Linear(x))       # [B,N,1] 复杂度分数
        h_hyp= lorentz_exp_origin(h, c) # 先进 Lorentz [B,N,d_h+1]
        x0'  = x0 * (1 + α*κ)           # Lorentz 内缩放 x0（κ大→边界），保流形约束
        out  = out_proj(log(h_hyp')) + x # log 回来 + 残差

    注意：径向校正必须在 Lorentz 内（exp 之后）做，不能先在欧氏调半径再
    exp/log——否则 exp/log 首尾恒等、双曲失效（实测差值 1e-7）。
    """

    def __init__(self, channels: int, hidden_dim: int = 64, curvature_init: float = 0.4):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        c_init = math.log(math.expm1(curvature_init))
        self.curvature = nn.Parameter(torch.tensor(c_init))
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.complexity = nn.Linear(channels, 1)          # 复杂度分数 κ
        self.out_proj = nn.Linear(hidden_dim, channels)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # α 用 softplus 参数化保证恒正（α≈0.1）；径向校正在 Lorentz 流形内
        # 缩放 x0，无需独立 r0（x0 本身由 h 的范数决定）
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))

    def _curvature(self) -> torch.Tensor:
        return F.softplus(self.curvature) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """曲率引导的径向校正。

        Args:
            x: [B, N, C] 特征（embed 后、stage 前）
        Returns:
            [B, N, C] 残差校正后的特征
        """
        c = self._curvature()
        h = self.layer_norm(self.in_proj(x))              # [B, N, d_h]
        kappa = torch.sigmoid(self.complexity(x))         # [B, N, 1] 复杂度分数
        alpha = F.softplus(self.alpha_raw)                # 恒正
        # 进 Lorentz 流形后做径向校正（不能在欧氏调半径再 exp/log，否则 exp/log
        # 首尾恒等、双曲失效——已验证 h_euc≈v 差值 1e-7）
        h_hyp = lorentz_exp_origin(h, c)                  # [B, N, d_h+1]
        x0 = h_hyp[..., 0:1]                              # 时间分量（径向位置）
        x_rest = h_hyp[..., 1:]
        x0_new = x0 * (1.0 + alpha * kappa)               # 高复杂度 → x0 大 → 边界
        # 保持流形约束 x0² - ‖x_rest‖² = 1/c
        scale = torch.sqrt((x0_new ** 2 - 1.0 / c).clamp_min(1e-6)
                           / (x0 ** 2 - 1.0 / c).clamp_min(1e-6))
        h_hyp_rect = torch.cat([x0_new, x_rest * scale], dim=-1)
        h_euc = lorentz_log_origin(h_hyp_rect, c)         # [B, N, d_h]
        out = self.out_proj(h_euc)                        # [B, N, C]
        return x + out                                    # 残差


class PolarFeatureLayer(nn.Module):
    """方向/半径分离特征层（Polar FC 思路，arXiv 2609.32899，欧氏对照版）。

    核心：传统 Linear 输出半径由 ‖Wx+b‖ 决定、不可独立控制，大半径处 Lorentz
    内积数值不稳（论文 A 节）。本组件把特征分解为方向（单位向量）与半径（标量），
    分别可学习映射再合回，半径可独立学。

    与 HyperbolicRadialRectify（chr，在双曲流形内做径向校正）形成「欧氏 vs 双曲」
    对照：两者同挂 embed 后，配对差可回答「双曲流形相对欧氏方向/半径分离的
    增量」，呼应 Hyper² 对「双曲是否装饰」的审计。

    机制（与 chr 结构镜像，唯一变量=是否进双曲；残差）：
        h    = LayerNorm(in_proj(x))     # [B,N,d_h]
        κ    = sigmoid(complexity(x))    # [B,N,1]（同 chr）
        h'   = h * (1 + α*κ)             # 欧氏径向校正（对应 chr 的 Lorentz x0 缩放）
        out  = out_proj(h')              # [B,N,C]
        return x + out                   # 残差

    严格对照 chr：chr = exp → [Lorentz 内缩放 x0 + 流形约束] → log；
    本组件 = [欧氏缩放范数]。除 exp/log 与 curvature 外结构逐项对齐。
    """

    def __init__(self, channels: int, hidden_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        # 与 chr 结构镜像（同 in_proj/complexity/alpha_raw），只是不进双曲
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.complexity = nn.Linear(channels, 1)
        self.out_proj = nn.Linear(hidden_dim, channels)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(self.in_proj(x))              # [B, N, d_h]
        kappa = torch.sigmoid(self.complexity(x))         # [B, N, 1] 复杂度分数
        alpha = F.softplus(self.alpha_raw)                # 恒正
        h_rect = h * (1.0 + alpha * kappa)                # 欧氏径向校正（对应 chr 的 x0 缩放）
        out = self.out_proj(h_rect)                       # [B, N, C]
        return x + out                                    # 残差


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
                 use_remote_g16: bool = False,
                 g16_treatment: str = "",
                 g16_rng_seed: int = 42,
                 use_rp1_selector: bool = False,
                 use_hyperbolic_selector: bool = False,
                 use_hyperbolic_classifier: bool = False,
                 use_hyperbolic_fusion: bool = False,
                 use_hyperbolic_embedding: bool = False,
                 use_hyperbolic_attention: bool = False,
                 use_hyperbolic_mlp: bool = False,
                 use_hyperbolic_gate: bool = False,
                 use_fca_gate: bool = False,
                 use_hyperbolic_radial: bool = False,
                 use_polar_feature: bool = False,
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
        if use_remote_g16 and (use_rp1_selector or use_hyperbolic_selector):
            raise ValueError("remote_g16 与 rp1_selector/hyperbolic_selector 互斥（同层采样，只能选一个）")
        # 两个门控都是替换 gated fusion 的备选（同层互斥，类似 rp1/hsel）：
        # 同开时 forward 的 if/elif 会静默走 hyperbolic_gate，必须显式报错
        if use_hyperbolic_gate and use_fca_gate:
            raise ValueError("hyperbolic_gate 与 fca_gate 互斥（同层门控，只能选一个）")
        if use_hyperbolic_radial and use_polar_feature:
            raise ValueError("hyperbolic_radial 与 polar_feature 互斥（同层特征组件，只能选一个）")
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
        # 替换式双曲门控（共识③ 融合层 gated→双曲加权）：替换 gated fusion 的
        # 欧氏通道门控，门控信号改由双曲距离学出。默认关闭；hyper_gates 仅在
        # 开启时创建，group_conv/bn_global 参数保留但不被调用（state_dict 键稳定）
        self.use_hyperbolic_gate = use_hyperbolic_gate
        self.use_fca_gate = use_fca_gate
        self.use_hyperbolic_radial = use_hyperbolic_radial
        self.use_polar_feature = use_polar_feature
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
            if use_remote_g16:
                # 16 点远程选择（论文灵魂）：v2 的 LocalRemoteGrouper（含六处理分派），
                # g16_treatment 选六种选点准则之一（random/spatial_fixed/euc_far/
                # euc_near/hyp_far/rp1_scorer，空串=原 topk 双曲行为）
                local_grouper = LocalRemoteGrouperV2(self.feature_list[i], self.group_number[i],
                                                     kneighbors=24, remote_g=16,
                                                     use_xyz=True, normalize="center",
                                                     g16_treatment=g16_treatment,
                                                     g16_rng_seed=g16_rng_seed,
                                                     stage_idx=i)
            elif use_rp1_selector:
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
        if self.use_hyperbolic_gate:
            # 替换式双曲门控：每 stage 一个 HyperbolicGate（双曲距离门控），
            # 替换 gated fusion 的 global 分支 + sigmoid 通道门控
            self.hyper_gates = nn.ModuleList([
                HyperbolicGate(self.feature_list[i + 1])
                for i in range(self.stages)
            ])
        if self.use_fca_gate:
            # 频域通道门控（FcaNet 思路）：每 stage 一个 FcaGate，用多频率 DCT
            # 分量（非仅 DC 均值）压缩每个通道，学出逐通道门控
            self.fca_gates = nn.ModuleList([
                FcaGate(self.feature_list[i + 1], self.group_number[i])
                for i in range(self.stages)
            ])
        if self.use_hyperbolic_radial:
            # 曲率感知双曲径向校正（PointCHR 思路）：挂在 embed 后、stage 前，
            # 单模块（非每 stage），对每个点的特征做径向校正
            self.hyper_radial = HyperbolicRadialRectify(self.feature_list[0])
        if self.use_polar_feature:
            # 方向/半径分离（Polar FC 思路，欧氏对照）：同挂 embed 后，单模块
            self.polar_feature = PolarFeatureLayer(self.feature_list[0])

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
        if self.use_hyperbolic_radial:
            # 曲率感知双曲径向校正（PointCHR）：embed 后、stage 前做逐点径向校正
            x = self.hyper_radial(x)
        elif self.use_polar_feature:
            # 方向/半径分离（Polar FC，欧氏对照）：embed 后做方向/半径分离
            x = self.polar_feature(x)
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
            if self.use_hyperbolic_gate:
                # 替换式双曲门控（共识③ gated→双曲加权）：门控信号由双曲距离
                # 学出，替换原 global 分支（mean pool -> group_conv -> sigmoid）
                x = self.hyper_gates[i](x_local, x)
            elif self.use_fca_gate:
                # 频域通道门控（FcaNet 思路）：门控信号由多频率 DCT 分量学出
                x = self.fca_gates[i](x_local, x)
            else:
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
