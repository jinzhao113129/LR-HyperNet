"""
Local+Remote 邻域分组模块（LR-HyperNet 核心组件）

在原 LocalGrouper（FPS 采样 + 局部 KNN）基础上增加 Remote 分支：
对每个 FPS 中心点，在特征空间中排除局部邻居后，选取距离最远的 G 个
远程事件，经双曲聚合得到全局上下文特征，与局部特征拼接融合。

设计参考：
  - SECNet/FECNet LocalGrouper（models/modules.py）：Local 分支与其完全一致
  - hyperbolic_fecnet.py HyperbolicGroupAggregator：双曲组内聚合
    （本文件的双曲运算与其前 70 行逐行一致，按要求复制而非重新发明）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # 兼容两种导入上下文：从项目根目录包导入 / models 目录在 sys.path 中
    from models.modules import furthest_point_sample, index_points, square_distance
except ImportError:  # pragma: no cover
    from modules import furthest_point_sample, index_points, square_distance


# ============================================================================
# 双曲几何基础运算（复制自 hyperbolic_fecnet.py 前 70 行，勿改语义）
# ============================================================================

def clip_to_disk(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """把点投影回 Poincaré 球内部。"""
    max_norm = (1.0 / torch.sqrt(c)) - eps
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    cond = norm > max_norm
    scale = torch.where(cond, max_norm / norm, torch.ones_like(norm))
    return x * scale


def poincare_log_origin(y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对数映射（原点锚定）：双曲空间 → 切空间。"""
    norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(eps)
    arg = (torch.sqrt(c) * norm).clamp(max=1.0 - eps)
    coef = (2.0 / torch.sqrt(c)) * torch.arctanh(arg)
    return torch.where(norm <= eps, torch.zeros_like(y), coef * (y / norm))


def poincare_exp_origin(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """指数映射（原点锚定）：切空间 → 双曲空间。"""
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    coef = torch.tanh(torch.sqrt(c) * norm / 2.0) / (torch.sqrt(c) * norm)
    out = coef * v
    return torch.where(norm <= eps, torch.zeros_like(out), clip_to_disk(out, c))


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """双曲距离。x,y: [..., d]，返回 [...]。"""
    x_norm_sq = (x * x).sum(-1).clamp(max=(1.0 / c) - eps)
    y_norm_sq = (y * y).sum(-1).clamp(max=(1.0 / c) - eps)
    diff_sq = ((x - y) ** 2).sum(-1)
    z = 1 + 2 * c * diff_sq / ((1 - c * x_norm_sq) * (1 - c * y_norm_sq)).clamp_min(eps)
    z = z.clamp_min(1.0 + eps)
    arccosh_z = torch.log(z + torch.sqrt((z - 1.0) * (z + 1.0)))
    return (1.0 / torch.sqrt(c)) * arccosh_z


# ============================================================================
# Lorentz 模型运算（v3.1：替代 Poincaré，消除边界梯度尖峰）
# 约定：曲率 c > 0，Lorentz 内积 ⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ
# 流形 H^d_c = {x ∈ R^{d+1} : ⟨x,x⟩_L = -1/c, x₀ > 0}
# 原点 o = (1/√c, 0, ..., 0)，切空间 T_o = {(0, v₁,...,v_d)}
# ============================================================================

def lorentz_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentz 内积：⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ。x,y: [..., d+1]，返回 [...]。"""
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)


def lorentz_exp_origin(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-6,
                       norm_max: float = 10.0) -> torch.Tensor:
    """指数映射（原点锚定）：欧氏切向量 v ∈ R^d → Lorentz 点 ∈ R^{d+1}。

    exp_o(v) = (cosh(√c·||v||)/√c, sinh(√c·||v||)·v/(√c·||v||))

    数值保护（v3.1 修复，两层）：
      1) 溢出防护：float32 下 cosh(√c·||v||) 在 ||v||≈141 时溢出为 inf
         （c=0.4）。范数超 norm_max 的部分线性缩放到 norm_max。
      2) 精度防护：Lorentz 内积 -x0y0+Σxy 是大数相减，float32 下
         ||v||>~11 时误差就会淹没真实距离（量级 1/c≈2.5），因此
         norm_max 取 10（sinh(√0.4·10)≈272，坐标平方 ~1.7e5，
         相减误差 ~0.02，可接受）。
    输出再做流形投影兜底，保证 ⟨x,x⟩_L = -1/c 精确成立。
    """
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(norm_max / norm, max=1.0)      # 超限缩放，饱和处梯度为 0
    v_c = v * scale
    norm_c = norm * scale                              # = min(||v||, norm_max)
    sqrt_c = torch.sqrt(c)
    cosh_term = torch.cosh(sqrt_c * norm_c) / sqrt_c   # [..., 1]
    # 修正：x[1:] 维必须带 1/√c 因子（标准 Lorentz 指数映射），
    # 否则流形约束 x0²-||x[1:]||²=1/c 不成立，双曲距离系统性≈0、注意力退化
    sinh_term = torch.sinh(sqrt_c * norm_c) * v_c / (sqrt_c * norm_c)  # [..., d]
    out = torch.cat([cosh_term, sinh_term], dim=-1)    # [..., d+1]
    # 流形投影兜底：x0 ← sqrt(1/c + ||x[1:]||²)，防御浮点误差
    x_rest = out[..., 1:]
    x0 = torch.sqrt((1.0 / c) + (x_rest * x_rest).sum(-1, keepdim=True))
    return torch.cat([x0, x_rest], dim=-1)


def lorentz_log_origin(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对数映射（原点锚定）：Lorentz 点 x ∈ R^{d+1} → 欧氏切向量 ∈ R^d。

    log_o(x) = arccosh(√c·x₀) / √(c·x₀² - 1) · x[1:]
    """
    sqrt_c = torch.sqrt(c)
    x0 = x[..., 0:1].clamp_min(1.0 / sqrt_c + eps)  # 保证 x₀ ≥ 1/√c
    theta = torch.acosh((sqrt_c * x0).clamp_min(1.0 + eps))  # [..., 1]
    denom = torch.sqrt((c * x0 ** 2 - 1.0).clamp_min(eps))  # [..., 1]
    return (theta / denom) * x[..., 1:]  # [..., d]


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Lorentz 双曲距离。x,y: [..., d+1]，返回 [...]。

    d(x,y) = (1/√c)·arccosh(-c·⟨x,y⟩_L) = (1/√c)·arccosh(c·(x₀y₀ - Σxᵢyᵢ))
    """
    inner = lorentz_inner_product(x, y)  # [...], 负值（流形上两点内积 < 0）
    arg = (-c * inner).clamp_min(1.0 + eps)
    return (1.0 / torch.sqrt(c)) * torch.acosh(arg)


# ============================================================================
# 远程邻居选择器
# ============================================================================

class RemoteSelector(nn.Module):
    """远程邻居选择器（第一版：特征距离 top-k，不做 learned scorer）。

    对每个中心点，在特征空间中计算其到全部 N 个点的距离，把局部邻居
    （local_idx 中的 K 个点）的距离置为 -inf 排除，再选距离最大的 G 个
    点作为远程邻居。即：远程事件 = 与中心点特征最不相似的 G 个事件。

    显存优化：与原 LocalGrouper 的分块 KNN 同一策略——完整距离矩阵
    [B, S, N] 在 N=10240 时过大，按 N 分块保留块内 top-G（largest），
    候选并集上再取全局 top-G。正确性论证同分块 KNN：全局 top-G 的每个
    元素必然是其所在 chunk 内的 top-G。
    """

    def __init__(self, remote_g: int = 16, chunk: int = 1024,
                 anchor_lambda: float = 0.5):
        """
        Args:
            remote_g: 每个中心点选取的远程邻居数 G
            chunk: 分块大小（与 LocalGrouper 的 KNN 分块一致取 1024）
            anchor_lambda: 全局吸收节点（PVAG 启发）混合权重。
                混合距离 d = d_to_centroid + anchor_lambda * d_to_global_anchor，
                即远程点同时要求"与中心点特征差异大"且"与全局表征差异大"。
                为 0 时退化为原版（仅特征距离 top-k）。
        """
        super().__init__()
        self.remote_g = remote_g
        self.chunk = chunk
        self.anchor_lambda = anchor_lambda

    @staticmethod
    def _exclude_local(chunk_dist: torch.Tensor, local_idx: torch.Tensor,
                       start: int) -> torch.Tensor:
        """把落在本 chunk [start, start+c) 内的局部邻居距离置为 -inf。

        Args:
            chunk_dist: [B, S, c] 当前块的距离矩阵
            local_idx: [B, S, K] 局部邻居的全局索引
            start: 当前块在 N 维上的起始偏移
        """
        B, S, c = chunk_dist.shape
        offset = local_idx - start  # [B, S, K]
        valid = (offset >= 0) & (offset < c)
        # 无效邻居统一散射到第 c 列（末尾补的一列 -inf），避免越界
        safe_idx = torch.where(valid, offset, torch.full_like(offset, c))
        padded = torch.cat(
            [chunk_dist, chunk_dist.new_full((B, S, 1), float("-inf"))], dim=-1)
        padded.scatter_(2, safe_idx, float("-inf"))
        return padded[..., :c]

    def forward(self, centroid_feat: torch.Tensor, all_feat: torch.Tensor,
                local_idx: torch.Tensor,
                global_anchor: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            centroid_feat: [B, S, C] 中心点特征
            all_feat: [B, N, C] 全部点特征
            local_idx: [B, S, K] 局部邻居索引（待排除）
            global_anchor: [B, S, C] 或 None。全局吸收节点特征（PVAG 启发）；
                非 None 时混合距离 = 到中心点距离 + anchor_lambda * 到锚点距离。
        Returns:
            remote_idx: [B, S, G] 远程邻居索引（沿 G 维升序）
        """
        N = all_feat.shape[1]
        B, S, C = centroid_feat.shape
        G = self.remote_g
        if N < G + local_idx.shape[-1]:
            raise ValueError(
                f"N={N} 不足以同时容纳 K={local_idx.shape[-1]} 局部邻居与 G={G} 远程邻居")
        # 全局锚点扩展为 [B, S, C]（可学习单向量 -> 广播到每个中心点）
        if global_anchor is not None:
            if global_anchor.dim() == 1:
                global_anchor = global_anchor.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        # 选择只产生离散索引，无需梯度；no_grad 避免距离矩阵挂计算图浪费显存
        with torch.no_grad():
            topk_list = []
            for start in range(0, N, self.chunk):
                end = min(start + self.chunk, N)
                c = end - start
                # 尾块可能不足 G 个元素：块内取 min(G, c) 个，不足则整块全作候选
                k = min(G, c)
                chunk_dist = square_distance(centroid_feat, all_feat[:, start:end, :])
                if global_anchor is not None:
                    # PVAG 式全局吸收：混合"到中心点"与"到全局锚点"的距离
                    anchor_dist = square_distance(global_anchor, all_feat[:, start:end, :])
                    chunk_dist = chunk_dist + self.anchor_lambda * anchor_dist
                chunk_dist = self._exclude_local(chunk_dist, local_idx, start)
                # largest=True：选特征距离最远的 G 个（排除项已是 -inf 不会被选中）
                topk_list.append(chunk_dist.topk(k, dim=-1, largest=True)[1] + start)
            cand_idx = torch.cat(topk_list, dim=-1)  # [B, S, n_chunks*G]
            # 候选并集上重算距离，取全局 top-G
            cand_points = index_points(all_feat, cand_idx)  # [B, S, n_chunks*G, C]
            cand_dist = ((cand_points - centroid_feat.unsqueeze(-2)) ** 2).sum(-1)
            if global_anchor is not None:
                cand_anchor_dist = (
                    (cand_points - global_anchor.unsqueeze(-2)) ** 2).sum(-1)
                cand_dist = cand_dist + self.anchor_lambda * cand_anchor_dist
            final = cand_dist.topk(G, dim=-1, largest=True)[1]
            remote_idx = torch.gather(cand_idx, -1, final)
        return remote_idx.sort(dim=-1)[0]


# ============================================================================
# KDA 因果时间编码器（Kimi Delta Attention 式 gated delta rule）
# ============================================================================

class KDACausalEncoder(nn.Module):
    """KDA 因果时间编码器（gated delta rule 状态递推）。

    对局部组内 K 个邻居按时间戳升序排列，做 gated delta rule 状态递推：
        S_i = (I - β_i k_i k_i^T) · Diag(α_i) · S_{i-1} + β_i k_i v_i^T
    每个位置用查询 q 从状态读出（out_i = S_i^T q_i），得到携带组内时间
    因果序的逐点特征。替代 v2 的 sin/cos 位置编码（time_enc）：时间信息
    由递推状态隐式建模，无需显式位置编码（Kimi K3 用 KDA 替代 ROPE）。

    显存控制：状态为低维 [B, S, D, D]（D=16），K=24 步循环更新，峰值
    ~ O(B·S·D²)；输出投影回 channel 维，与 LocalGrouper 下游维度对齐。
    """

    def __init__(self, channel: int, state_dim: int = 16, bidirectional: bool = False):
        """
        Args:
            channel: 输入/输出特征维度 C
            state_dim: KDA 状态维度 D（k/v/q 投影维，默认 16）
            bidirectional: 是否双向因果递推（v3.1，HydraMamba 启发）。
                True 时前向+后向各跑一次，输出拼接后 Linear 融合，
                事件分类是离线任务，双向可利用完整时间上下文。
        """
        super().__init__()
        self.channel = channel
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.k_proj = nn.Linear(channel, state_dim)
        self.v_proj = nn.Linear(channel, state_dim)
        self.q_proj = nn.Linear(channel, state_dim)
        self.beta_proj = nn.Linear(channel, 1)       # 写入门（标量/组）
        self.alpha_proj = nn.Linear(channel, state_dim)  # 逐通道遗忘门
        self.out_proj = nn.Linear(state_dim, channel)
        self.layer_norm = nn.LayerNorm(channel)
        # 双向融合层（前向+后向 state_dim 拼接 → state_dim）
        if bidirectional:
            self.fuse_proj = nn.Linear(state_dim * 2, state_dim)
        # 初始化：β 写入门初始≈0（sigmoid(-5)≈0.0067，慢启动），
        # α 遗忘门初始=1（exp(0)，不遗忘），保证初期接近纯累积、稳定起步
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, -5.0)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.zeros_(self.alpha_proj.bias)

    def _causal_scan(self, k, v, q, beta, alpha):
        """gated delta rule 因果扫描（K 步递推），供前向/后向复用。

        Args:
            k,v,q: [B, S, K, D]，已按扫描顺序排列
            beta: [B, S, K, 1] 写入门
            alpha: [B, S, K, D] 遗忘门
        Returns:
            [B, S, K, D] 逐位置读出特征
        """
        B, S, K, D = k.shape
        state = k.new_zeros(B, S, D, D)
        outs = []
        for i in range(K):
            ki = k[:, :, i]
            vi = v[:, :, i]
            qi = q[:, :, i]
            betai = beta[:, :, i]
            alphai = alpha[:, :, i]
            kS = torch.einsum('bsi,bsij->bsj', ki, state)
            state = state * alphai.unsqueeze(-1) \
                - betai.unsqueeze(-1) * torch.einsum('bsi,bsj->bsij', ki, kS) \
                + betai.unsqueeze(-1) * torch.einsum('bsi,bsj->bsij', ki, vi)
            state_norm = state.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            state = state * torch.clamp(50.0 / state_norm, max=1.0)
            outs.append(torch.einsum('bsji,bsj->bsi', state, qi))
        return torch.stack(outs, dim=2)  # [B, S, K, D]

    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, K, C] 局部邻居特征（KNN 顺序，与 grouped_xytp 对应）
            ts: [B, S, K, 1] 邻居时间戳（xytp 通道 0）
        Returns:
            [B, S, K, C] 携带组内时间因果序的逐点特征（与 x 同形状）
        """
        # 1. 组内按时间戳升序排序（因果扫描顺序）
        order = ts[..., 0].argsort(dim=-1)                     # [B, S, K]
        x_sorted = torch.gather(x, 2, order.unsqueeze(-1).expand_as(x))
        # 2. 投影（前向/后向共享参数）
        k = self.k_proj(x_sorted)                              # [B, S, K, D]
        v = self.v_proj(x_sorted)
        q = self.q_proj(x_sorted)
        beta = torch.sigmoid(self.beta_proj(x_sorted)).clamp(max=0.5)
        alpha = torch.exp(self.alpha_proj(x_sorted)).clamp(min=0.05, max=1.0)
        # 3. 前向因果扫描（时间正序）
        out_fwd = self._causal_scan(k, v, q, beta, alpha)     # [B, S, K, D]
        if self.bidirectional:
            # 4. 后向因果扫描（时间逆序，沿 K 维翻转）
            out_bwd = self._causal_scan(
                k.flip(dims=[2]), v.flip(dims=[2]),
                q.flip(dims=[2]), beta.flip(dims=[2]),
                alpha.flip(dims=[2]))
            # 后向输出按原顺序翻回
            out_bwd = out_bwd.flip(dims=[2])
            # 5. 双向融合
            out = self.fuse_proj(torch.cat([out_fwd, out_bwd], dim=-1))
        else:
            out = out_fwd
        return self.layer_norm(self.out_proj(out)) * 0.1


# ============================================================================
# 远程邻域双曲聚合器
# ============================================================================

class HyperbolicRemoteAggregator(nn.Module):
    """远程邻域双曲聚合（核心逻辑复制自 hyperbolic_fecnet.HyperbolicGroupAggregator）。

    流程：指数映射到 Poincaré 球 → 以组内质心为锚计算双曲距离注意力 →
    切空间 Möbius 线性变换 → 注意力加权聚合 → 投影回欧氏空间 + 残差
    + LayerNorm。曲率可学习（abs + 1e-4 保证为正）。

    输入: [B, S, G, C]（G 个远程事件的特征）
    输出: [B, S, C]（双曲质心聚合后的特征）
    """

    def __init__(self, hidden_size: int, curvature_init: float = 0.4,
                 temperature: float = 1.0, hyperbolic_type: str = "poincare"):
        """
        Args:
            hidden_size: 特征维度 C
            curvature_init: 双曲曲率初始值（论文 c ∈ {0.2, 0.4, 0.6, 0.8}）
            temperature: 注意力温度系数
            hyperbolic_type: "poincare"（庞加莱球，原版）或 "lorentz"（洛伦兹模型，v3.1 新增，消除边界梯度尖峰）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.curvature = nn.Parameter(torch.tensor(curvature_init))
        self.temperature = temperature
        self.hyperbolic_type = hyperbolic_type.lower()
        assert self.hyperbolic_type in ["poincare", "lorentz"], \
            f"hyperbolic_type must be 'poincare' or 'lorentz', got {hyperbolic_type}"
        # Möbius 线性变换：切空间 W·log_o(x) + b 再指数映射回双曲（等价 W ⊗_c x ⊕_c b）
        self.mobius_linear = nn.Linear(hidden_size, hidden_size)
        self.mobius_bias = nn.Parameter(torch.zeros(hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _curvature(self) -> torch.Tensor:
        """曲率需为正，用 abs+eps 保证（curvature_init=0.4 时实际曲率≈0.4）。"""
        return torch.abs(self.curvature) + 1e-4

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """单个组的双曲聚合。

        v3.1 支持两种双曲空间：
          - poincare：庞加莱球（原版，边界 atanh 有梯度尖峰风险）
          - lorentz：洛伦兹模型（无边界问题，数值更稳定）

        Args:
            x: [M, K, C]，K 个点的欧氏特征
        Returns:
            [M, C]，聚合后的欧氏特征
        """
        c = self._curvature()
        if self.hyperbolic_type == "poincare":
            # 原版 Poincaré 流程（与 HyperbolicGroupAggregator.forward 逐行一致）
            x_hyp = poincare_exp_origin(x, c)
            centroid_euclidean = x.mean(dim=1, keepdim=True)
            anchor = poincare_exp_origin(centroid_euclidean, c)
            distances = poincare_distance(anchor.expand_as(x_hyp), x_hyp, c)
            attn_weights = F.softmax(-distances / self.temperature, dim=-1)
            x_tangent = poincare_log_origin(x_hyp, c)
            x_transformed = self.mobius_linear(x_tangent) + self.mobius_bias
            x_hyp_transformed = poincare_exp_origin(x_transformed, c)
            agg_tangent = torch.bmm(
                attn_weights.unsqueeze(1), poincare_log_origin(x_hyp_transformed, c)
            ).squeeze(1)
            agg_hyp = poincare_exp_origin(agg_tangent, c)
            agg_euclidean = poincare_log_origin(agg_hyp, c)
        else:  # lorentz
            # Lorentz 模型流程（无边界梯度尖峰）
            x_hyp = lorentz_exp_origin(x, c)  # [M, K, C+1]
            centroid_euclidean = x.mean(dim=1, keepdim=True)  # [M, 1, C]
            anchor = lorentz_exp_origin(centroid_euclidean, c)  # [M, 1, C+1]
            distances = lorentz_distance(anchor.expand_as(x_hyp), x_hyp, c)  # [M, K]
            attn_weights = F.softmax(-distances / self.temperature, dim=-1)  # [M, K]
            # Möbius 线性变换（切空间）
            x_tangent = lorentz_log_origin(x_hyp, c)  # [M, K, C]
            x_transformed = self.mobius_linear(x_tangent) + self.mobius_bias  # [M, K, C]
            x_hyp_transformed = lorentz_exp_origin(x_transformed, c)  # [M, K, C+1]
            # 切空间注意力加权聚合
            agg_tangent = torch.bmm(
                attn_weights.unsqueeze(1), lorentz_log_origin(x_hyp_transformed, c)
            ).squeeze(1)  # [M, C]
            # 映射回 Lorentz 再回到欧氏
            agg_hyp = lorentz_exp_origin(agg_tangent, c)  # [M, C+1]
            agg_euclidean = lorentz_log_origin(agg_hyp, c)  # [M, C]
        # 输出投影 + 残差 + LayerNorm（残差保证梯度流畅）
        out = self.out_proj(agg_euclidean)
        residual = x.mean(dim=1)
        return self.layer_norm(out + residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, G, C]，G 个远程事件的特征
        Returns:
            [B, S, C]，聚合后的特征
        """
        B, S, G, C = x.shape
        out = self._aggregate(x.reshape(-1, G, C))  # [B*S, C]
        return out.reshape(B, S, C)




class EuclideanRemoteAggregator(nn.Module):
    """远程邻域欧氏聚合（双曲版的等价对照，消融实验用）。

    与 HyperbolicRemoteAggregator 逐项同构：同 anchor（组内欧氏质心）、
    同变换-注意力加权聚合-out_proj+残差+LayerNorm 结构，
    唯一区别：距离用欧氏平方距离替代双曲距离，
    温度为可学习标量（替代可学习曲率）。参数量与双曲版完全一致。

    输入: [B, S, G, C]（G 个远程事件的特征）
    输出: [B, S, C]（欧氏质心聚合后的特征）
    """

    def __init__(self, hidden_size: int, temperature_init: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.mobius_linear = nn.Linear(hidden_size, hidden_size)
        self.mobius_bias = nn.Parameter(torch.zeros(hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _temperature(self) -> torch.Tensor:
        """正温度 T = abs(T_param) + 1e-4"""
        return torch.abs(self.temperature) + 1e-4

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """单个组的欧氏聚合。

        Args:
            x: [M, K, C]，K 个点的欧氏特征
        Returns:
            [M, C]，聚合后的欧氏特征
        """
        T = self._temperature()
        centroid = x.mean(dim=1, keepdim=True)
        distances = ((x - centroid) ** 2).sum(dim=-1)
        attn_weights = F.softmax(-distances / T, dim=-1)
        x_transformed = self.mobius_linear(x) + self.mobius_bias
        agg = torch.bmm(attn_weights.unsqueeze(1), x_transformed).squeeze(1)
        out = self.out_proj(agg)
        residual = x.mean(dim=1)
        return self.layer_norm(out + residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, G, C = x.shape
        out = self._aggregate(x.reshape(-1, G, C))
        return out.reshape(B, S, C)


# ============================================================================
# Local+Remote 分组器
# ============================================================================

class LocalRemoteGrouper(nn.Module):
    """Local+Remote 邻域分组器（替换原 LocalGrouper）。

    与原 LocalGrouper 的区别仅在输出特征的拼接方式：
      - 原来：cat(局部归一化特征 [C+4], 中心点特征 [C]) → [K, 2C+4]
      - 现在：cat(局部归一化特征 [C+4], 远程双曲聚合特征 [C]) → [K, 2C+4]
    即远程双曲特征替换原来重复拼接的中心点特征，输出维度与原 LocalGrouper
    完全一致（2C+4），下游 FFT/聚合模块无需改动，且形状上兼容 FECNet 预训练权重。

    FPS 采样、局部 KNN（分块显存优化）、center/anchor 归一化、new_xytp 的
    计算（局部邻居 xytp 均值）均与原 LocalGrouper 完全一致。
    """

    def __init__(self, channel: int, groups: int, kneighbors: int = 24,
                 remote_g: int = 16, use_xyz: bool = True,
                 normalize: str = "center", curvature_init: float = 0.4,
                 use_global_anchor: bool = False, anchor_lambda: float = 0.5,
                 kda: bool = False, kda_bidirectional: bool = False,
                 hyperbolic_type: str = "poincare", **kwargs):
        """
        Args:
            channel: 输入特征维度 C
            groups: FPS 采样组数 S
            kneighbors: 局部邻居数 K
            remote_g: 远程邻居数 G
            use_xyz: 归一化 anchor 模式下是否拼接坐标
            normalize: "center" / "anchor" / None，与原 LocalGrouper 语义一致
            curvature_init: 双曲聚合器曲率初值
            use_global_anchor: 是否启用全局吸收节点（PVAG 启发）。
                启用时为该 stage 维护一个可学习全局锚点向量 [C]，
                远程选择混合"到中心点距离 + anchor_lambda*到锚点距离"。
            anchor_lambda: 全局锚点混合权重（见 RemoteSelector）
            kda: 是否启用 KDA 因果时间编码（Kimi Delta Attention 式 gated
                delta rule）。启用时局部邻居特征按组内时间戳因果扫描，
                替代 v2 的 sin/cos 位置编码（time_enc），输出维度不变 [C]。
            kda_bidirectional: KDA 是否双向因果递推（v3.1，HydraMamba 启发）。
                仅 kda=True 时生效，前向+后向各跑一次后融合。
            hyperbolic_type: "poincare"（庞加莱球，原版）或 "lorentz"
                （洛伦兹模型，v3.1 新增，消除边界梯度尖峰）
        """
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.fps_parameter = nn.Parameter(torch.ones([1, 1, 4]))
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter ({self.normalize}), set to None. "
                  f"Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 4 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))
        # KDA 因果时间编码（可选）：局部特征按时间因果扫描，替代 time_enc
        self.kda_enc = KDACausalEncoder(channel, bidirectional=kda_bidirectional) if kda else None
        # Remote 分支：远程邻居选择 + 双曲聚合
        self.remote_selector = RemoteSelector(remote_g=remote_g,
                                              anchor_lambda=anchor_lambda)
        if hyperbolic_type == "euclidean":
            self.hyper_agg = EuclideanRemoteAggregator(channel, temperature_init=1.0)
        else:
            self.hyper_agg = HyperbolicRemoteAggregator(channel, curvature_init=curvature_init,
                                                         hyperbolic_type=hyperbolic_type)
        # 全局吸收节点（PVAG 启发）：每 stage 一个可学习锚点向量
        self.use_global_anchor = use_global_anchor
        if use_global_anchor:
            self.global_anchor = nn.Parameter(torch.zeros(channel))

    def _knn_idx(self, new_points: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """分块 KNN（与原 LocalGrouper 的显存优化实现完全一致）。

        Args:
            new_points: [B, S, C] 中心点特征
            points: [B, N, C] 全部点特征
        Returns:
            idx: [B, S, K] 每个中心点的 K 个最近邻索引（升序）
        """
        N = points.shape[1]
        knn_k = self.kneighbors
        topk_list = []
        for start in range(0, N, 1024):
            end = min(start + 1024, N)
            chunk_dist = square_distance(new_points, points[:, start:end, :])  # [B, S, c]
            topk_list.append(chunk_dist.topk(knn_k, dim=-1, largest=False)[1] + start)
        cand_idx = torch.cat(topk_list, dim=-1)  # [B, S, n_chunks*k]
        cand_points = index_points(points, cand_idx)  # [B, S, n_chunks*k, C]
        cand_dist = ((cand_points - new_points.unsqueeze(-2)) ** 2).sum(-1)
        final_local = cand_dist.topk(knn_k, dim=-1, largest=False)[1]
        idx = torch.gather(cand_idx, -1, final_local)
        return idx.sort(dim=-1)[0]

    def forward(self, xytp: torch.Tensor, points: torch.Tensor):
        """
        Args:
            xytp: [B, N, 4] 事件坐标（x, y, t, p）
            points: [B, N, C] 事件特征
        Returns:
            new_xytp: [B, S, 4] 下一 stage 的坐标（局部邻居 xytp 均值，与原来一致）
            new_points: [B, S, K, 2C+4] 局部归一化特征 concat 远程双曲聚合特征
        """
        B, N, C = xytp.shape
        S = self.groups
        # ---- FPS 采样（与原 LocalGrouper 完全一致）----
        xytp = self.fps_parameter * xytp
        xytp = xytp.contiguous()
        fps_idx = furthest_point_sample(xytp[:, :, 0:3], self.groups).long()
        fps_idx, _ = torch.sort(fps_idx, dim=1)
        centroid_xytp = index_points(xytp, fps_idx)     # [B, S, 4]
        centroid_feat = index_points(points, fps_idx)   # [B, S, C]
        # ---- Local 分支：特征空间 KNN + 归一化拼接（与原 LocalGrouper 语义一致）----
        idx = self._knn_idx(centroid_feat, points)      # [B, S, K]
        grouped_xytp = index_points(xytp, idx)          # [B, S, K, 4]
        new_xytp = torch.mean(grouped_xytp, dim=2)      # [B, S, 4]
        if self.kda_enc is not None:
            # KDA 因果时间编码：残差连接（原始特征 + KDA 时间编码增量），
            # 防止 KDA 内部递推崩溃时局部特征完全失效（v3 epoch8 静默崩溃的根因）
            raw_local = index_points(points, idx)
            local_feat = raw_local + self.kda_enc(raw_local, grouped_xytp[..., 0:1])
        else:
            local_feat = index_points(points, idx)
        grouped_points = torch.cat([local_feat, grouped_xytp], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:  # anchor
                mean = torch.cat([centroid_feat, new_xytp], dim=-1) if self.use_xyz else centroid_feat
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1,
                            keepdim=True).unsqueeze(-1).unsqueeze(-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
        # ---- Remote 分支：排除局部邻居 → 特征距离最远 G 点 → 双曲聚合 ----
        anchor = None
        if self.use_global_anchor:
            anchor = self.global_anchor  # [C]，RemoteSelector 内部广播到 [B,S,C]
        remote_idx = self.remote_selector(centroid_feat, points, idx, anchor)  # [B, S, G]
        remote_points = index_points(points, remote_idx)               # [B, S, G, C]
        remote_feat = self.hyper_agg(remote_points)                    # [B, S, C]
        if self.use_global_anchor:
            # 全局吸收节点注入（PVAG 思路，保证可导）：
            # 远程选择（离散索引）本身不可导，锚点梯度经此处加法路径回流，
            # 学到"全局代表性方向"，作为双曲聚合输出的全局偏置。
            remote_feat = remote_feat + self.global_anchor.unsqueeze(0).unsqueeze(0)
        # ---- 融合：远程双曲特征替换原中心点特征的重复拼接，广播到 K 维 ----
        remote_feat = remote_feat.unsqueeze(2).expand(-1, -1, self.kneighbors, -1)
        new_points = torch.cat([grouped_points, remote_feat], dim=-1)  # [B, S, K, 2C+4]
        return new_xytp, new_points


# ============================================================================
# 双曲原型分类头（路线A：双曲从聚合层搬到分类头）
# ============================================================================
class HyperbolicPrototypeClassifier(nn.Module):
    """双曲原型分类器：欧氏特征 → Lorentz 空间 → 与类原型距离作为 logits。

    设计动机：双曲增益在文献中最常出现在嵌入/分类层，而非低层聚合的
    注意力距离。N-Caltech101 类别有 WordNet 层级结构，双曲原型能利用
    层级先验。

    前向：
      1. 欧氏特征 x ∈ R^D 经线性层映射到切空间 v ∈ R^d
      2. lorentz_exp_origin(v) → Lorentz 点 ∈ R^{d+1}
      3. 可学习类原型 prototypes ∈ R^{C×d} 映射到 Lorentz 空间
      4. logits = -lorentz_distance(x_lorentz, proto_lorentz)  [B, C]

    Args:
        input_dim: 输入欧氏特征维度（如 1080 = 540*2）
        num_classes: 类别数（101）
        embed_dim: 双曲嵌入维度（默认 64）
        curvature_init: 曲率初始值（默认 0.4）
    """

    def __init__(self, input_dim: int, num_classes: int,
                 embed_dim: int = 64, curvature_init: float = 0.4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fc = nn.Linear(input_dim, embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.1)
        self.curvature = nn.Parameter(torch.tensor(curvature_init))

    def _curvature(self) -> torch.Tensor:
        """正曲率 c = |c_param| + 1e-4"""
        return torch.abs(self.curvature) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] 欧氏特征（backbone 输出 mean+max 拼接）
        Returns:
            [B, num_classes] 分类 logits（负双曲距离）
        """
        c = self._curvature()
        v = self.fc(x)  # [B, embed_dim]
        x_lorentz = lorentz_exp_origin(v, c)  # [B, embed_dim+1]
        proto_lorentz = lorentz_exp_origin(self.prototypes, c)  # [C, embed_dim+1]
        dist = lorentz_distance(
            x_lorentz.unsqueeze(1),
            proto_lorentz.unsqueeze(0),
            c
        )  # [B, C]
        return -dist
