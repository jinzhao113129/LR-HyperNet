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

import math

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

def stk_soft_topk(scores, k, delta=1e-3):
    """ST_k 软权重（NeurIPS 2024，learned scorer 的可微 top-k）。

    w_i = SReLU'(s_i - λ) = 0.5 * (1 + (s_i-λ)/sqrt((s_i-λ)^2 + δ^2))
    λ = 第 k 大元素（非可微 kthvalue + detach，每次 forward 重算），Σw ≈ k。
    Codex 复审：λ 不是可训练参数；δ 需与 score 尺度匹配（先归一化 scorer 分数）。
    """
    lam = scores.detach().kthvalue(scores.shape[-1] - k + 1, dim=-1, keepdim=True).values
    x = scores - lam
    return 0.5 * (1 + x / torch.sqrt(x * x + delta * delta))


def gumbel_ste_topk(scores, k, tau=1.0):
    """STE + Gumbel-softmax（学长推荐，learned scorer 可微 top-k 备选）。

    前向：硬 top-k one-hot（质量集中精确，Σw=k）；反向：Gumbel-softmax 软梯度
    （稠密，所有点都有梯度，scorer 学得快，解决 ST_k 稀疏梯度冷启动慢的问题）。
    """
    topk = torch.topk(scores, k, dim=-1).indices  # [B,S,k]
    hard = torch.zeros_like(scores).scatter_(-1, topk, 1.0)  # 前向硬 one-hot
    gumbel = -torch.log(-torch.log(torch.rand_like(scores).clamp(min=1e-8)))
    soft = torch.softmax((scores + gumbel) / tau, dim=-1) * k  # 反向稠密梯度
    return hard - soft.detach() + soft


def lapsum_soft_topk(scores, k, alpha=0.5):
    """LapSum 软 top-k（LapSum 2503.06242）：拉普拉斯 CDF 软选择，Σp=k 精确。

    p_i = F((s_i − threshold)/α)，F 标准拉普拉斯 CDF；threshold 是 F-Sum(x)=n−k 的解
    （第 k 大元素的平滑近似），二分搜索求逆。Σp=k 精确、全可微、确定性（无噪声）。
    相比 Gumbel STE（前向 one-hot 近似）和 ST_k（稀疏梯度），LapSum 是最优：精确+稠密+确定。
    """
    n = scores.shape[-1]

    def laplace_cdf(x):
        x = torch.clamp(x, -30, 30)  # 防 exp 溢出
        return torch.where(x < 0, 0.5 * torch.exp(x), 1 - 0.5 * torch.exp(-x))

    with torch.no_grad():
        # 二分求 threshold 放 no_grad：只保留 p_i 的对角梯度 ∂p_i/∂s_i（Codex P1），
        # 避免 40 层二分 autograd 图 + τ 随 score 移动的二阶梯度。
        lo = scores.min(dim=-1, keepdim=True).values - 5 * alpha
        hi = scores.max(dim=-1, keepdim=True).values + 5 * alpha
        target = n - k  # 第 k 大 = 第 (n-k) 小
        for _ in range(40):
            mid = (lo + hi) / 2
            fsum = laplace_cdf((mid - scores) / alpha).sum(dim=-1, keepdim=True)
            hi = torch.where(fsum > target, mid, hi)
            lo = torch.where(fsum <= target, mid, lo)
        threshold = (lo + hi) / 2
    return laplace_cdf((scores - threshold) / alpha)


def eot_soft_topk(scores, k, tau=0.5, iters=20):
    """熵正则最优传输的软 top-k（SOFT 2002.06504 / CS-Net 2501.10789）。

    把「选 k 个」建成 2 列 OT：列 0 = 选中、列 1 = 未选。
    行边际 1/M，列边际 [k/M, (M−k)/M]；代价 C=[−s, 0]（列常数不影响传输计划）。
    对数域 Sinkhorn 求计划 P，返回 w = P[:,0]·M（Σw=k，全可微、确定性）。

    Args:
        scores: [..., M] 贡献分数
    Returns:
        w: [..., M]，Σw = k
    """
    M = scores.shape[-1]
    cost = torch.stack([-scores, torch.zeros_like(scores)], dim=-1)  # [...,M,2]
    log_mu = scores.new_full(scores.shape, -math.log(M))             # [...,M]
    log_nu = torch.log(scores.new_tensor([k / M, (M - k) / M]))      # [2]
    f = torch.zeros_like(scores)                                     # [...,M]
    g = torch.zeros_like(cost)                                       # [...,M,2]
    for _ in range(iters):
        f = tau * (log_mu - torch.logsumexp((g - cost) / tau, dim=-1))
        # g 的边际只在 2 列上；算完 [...,2] 后补回 [...,1,2] 以便与 cost 广播
        g = tau * (log_nu - torch.logsumexp(
            (f.unsqueeze(-1) - cost) / tau, dim=-2)).unsqueeze(-2)
    # P 的元素是传输质量（≤1），log 域截断防数值溢出
    log_p = ((f.unsqueeze(-1) + g - cost) / tau).clamp(min=-30.0, max=0.0)
    return torch.exp(log_p)[..., 0] * M


class RemoteSelector(nn.Module):
    """远程邻居选择器（第一版：特征距离 top-k，不做 learned scorer）。

    对每个中心点，在特征空间中计算其到全部 N 个点的距离，把局部邻居
    （local_idx 中的 K 个点）的距离置为 -inf 排除，再选距离最大的 G 个
    点作为远程邻居。即：远程事件 = 与中心点特征最不相似的 G 个事件。

    显存优化：与原 LocalGrouper 的分块 KNN 同一策略——完整距离矩阵
    [B, S, N] 在 N=10240 时过大，按 N 分块保留块内 top-G（largest），
    候选并集上再取全局 top-G。正确性论证同分块 KNN：全局 top-G 的每个
    元素必然是其所在 chunk 内的 top-G。

    selector_mode（RP1 论文的选点模式消融）：
      - "topk"：特征距离 top-G（默认，上述行为）
      - "greedy_div"：贪心 max-min 多样性选择
      - "random"：候选并集中随机采样 G 个（RP1 随机基线，不用距离打分）
      - "spatial_uniform"：x,y 空间均匀网格采样（RP1 最好的 cheap selector）
    后两种模式不计算 cand_dist，块内候选也改为随机收集（不依赖特征距离）。
    """

    def __init__(self, remote_g: int = 16, chunk: int = 1024,
                 anchor_lambda: float = 0.5,
                 use_hyperbolic: bool = False,
                 curvature_init: float = 0.4,
                 hyperbolic_type: str = "lorentz",
                 use_rp1_selector: bool = False,
                 rp1_lambda: float = 1.0,
                 selector_mode: str = "topk",
                 div_lambda: float = 1.0,
                 # ==== G16 六处理分派（2026-10-15, Codex 审查版）====
                 g16_treatment: str = "",
                 stage_idx: int = 0,
                 rng_seed: int = 42,
                 # ==== learned scorer（core idea 1，2026-10-24 Codex 复审）====
                 channel: int = 64,
                 learned_M: int = 256,
                 cs_chunk: int = 256,
                 learned_delta: float = 1e-3):
        """
        Args:
            remote_g: 每个中心点选取的远程邻居数 G
            chunk: 分块大小（与 LocalGrouper 的 KNN 分块一致取 1024）
            anchor_lambda: 全局吸收节点（PVAG 启发）混合权重。
                混合距离 d = d_to_centroid + anchor_lambda * d_to_global_anchor，
                即远程点同时要求"与中心点特征差异大"且"与全局表征差异大"。
                为 0 时退化为原版（仅特征距离 top-k）。
            use_hyperbolic: 是否用双曲距离选点（学长建议：16个点在双曲空间里找）。
                True 时先 exp map 到双曲空间，再用双曲距离代替欧氏距离。
            curvature_init: 双曲曲率初始值（仅 use_hyperbolic=True 时有效）
            hyperbolic_type: "poincare" 或 "lorentz"（仅 use_hyperbolic=True 时有效）
            use_rp1_selector: 是否启用 RP1 空间新颖性选择器（消融实验）。
                True 时远程选择得分 = 特征距离 + rp1_lambda * 空间新颖性
                （空间新颖性 = 候选点与局部邻居的最小欧氏距离），
                即优先选"特征不相似且空间位置新颖"的远程点（RP1 论文：
                Oracle 68% vs 简单 KNN 61%）。False 时保持原行为不变。
            rp1_lambda: 空间新颖性权重（仅 use_rp1_selector=True 时有效）
            selector_mode: 远程点最终选择策略。
                "topk"：原版独立 top-G 打分（默认，行为完全不变）；
                "greedy_div"：贪心 max-min 多样性选择——第 1 个点取得分
                最高的，后续每个点选 "得分 + div_lambda * (与已选集合的
                最小特征距离)" 最大的，避免 G 个远程点全落在同一特征
                方向（独立 top-k 的冗余问题）；
                "random"：RP1 论文的随机选择基线——在候选并集 cand_idx
                （分块阶段已排除局部邻居）中随机采样 G 个，不用距离打分；
                "spatial_uniform"：RP1 论文最好的 cheap selector——把
                x,y ∈ [0,1] 均分为 grid_size×grid_size 网格（grid_size =
                ceil(√G)，G=16 时 4×4=16 格正好每格 1 点），每个网格内
                随机选 1 个候选点，按随机网格访问序凑够 G 个（非空网格
                不足时随机补齐），在原始空间而非特征空间保证覆盖均匀。
            div_lambda: 多样性项权重（仅 selector_mode="greedy_div" 时有效）。
                为 0 时贪心退化为逐个 argmax，等价于 top-k 排序结果，
                完全回退原行为。
        """
        super().__init__()
        self.remote_g = remote_g
        self.chunk = chunk
        self.anchor_lambda = anchor_lambda
        # ==== Codex 第四轮 P1-2: treatment × 旧 flag 互斥断言 ====
        # 六处理是纯度实验：--g16_treatment=euc_far --use_hyperbolic_selector
        # 会实际跑成双曲选点，但 log/checkpoint 仍叫 euc_far——静默叠加必须
        # 显式拦截。合法组合：hyp_far 才允许 use_hyperbolic；
        # rp1_scorer 才允许 use_rp1_selector；空 treatment 不受任何限制。
        if g16_treatment:
            if use_hyperbolic and g16_treatment not in ('hyp_far', 'hyp_near'):
                raise ValueError(
                    f"g16_treatment='{g16_treatment}' 与 use_hyperbolic_selector=True "
                    f"冲突：六处理必须纯净——只有 hyp_far/hyp_near 走双曲选点。"
                    f"请去掉 --use_hyperbolic_selector。")
            if use_rp1_selector and g16_treatment != 'rp1_scorer':
                raise ValueError(
                    f"g16_treatment='{g16_treatment}' 与 use_rp1_selector=True "
                    f"冲突：六处理必须纯净——只有 rp1_scorer 走 RP1 空间新颖性。"
                    f"请去掉 --use_rp1_selector。")
        # ==== P0-1 修复（Codex）: treatment 映射提前到 buffer 注册之前 ====
        # hyp_far 需要 register_buffer("c")——原版映射在 L225 而 buffer
        # 注册在 L252 检查构造参数 use_hyperbolic，导致 self.use_hyperbolic
        # 虽被覆写为 True 但 c 没注册（AttributeError: no attribute 'c'）。
        # 修法: g16_treatment 的 use_hyperbolic 映射在这里（最早处）完成。
        if g16_treatment in ('hyp_far', 'hyp_near'):
            use_hyperbolic = True  # 覆写构造参数——buffer 注册将看到 True
            if hyperbolic_type.lower() == 'poincare':
                hyperbolic_type = 'lorentz'  # v3.1 推荐（数值稳定）
        self.use_hyperbolic = use_hyperbolic
        self.curvature_init = curvature_init
        self.hyperbolic_type = hyperbolic_type.lower()
        # RP1 空间新颖性选择器（消融实验）
        self.use_rp1_selector = use_rp1_selector
        self.rp1_lambda = rp1_lambda
        # cosine 相似度选点（cosine_near treatment，学长 Global Feature-KNN 口径）
        self.use_cosine = False
        # 远程点选择模式：topk / greedy_div / random / spatial_uniform
        assert selector_mode in ("topk", "greedy_div", "random", "spatial_uniform"), \
            f"selector_mode must be one of 'topk'/'greedy_div'/'random'/'spatial_uniform', got {selector_mode}"
        self.selector_mode = selector_mode
        self.div_lambda = div_lambda
        # ==== G16 六处理分派状态（空 treatment = 原行为完全不动）====
        # 六处理显式名（Codex 清单#1）: random / spatial_fixed / euc_far /
        # euc_near / hyp_far / hyp_near / rp1_scorer。非空时激活分派+缓存生产。
        self.g16_treatment = g16_treatment
        self.stage_idx = stage_idx
        # RNG 隔离（Codex 清单#4）: 每实例独立 Generator，
        # seed = rng_seed*1000 + stage_idx（写进 checkpoint 的 args 恢复）
        self._g16_rng_seed = rng_seed * 1000 + stage_idx
        # ==== 六处理真分派（Codex P0-2: treatment 映射到内部开关）====
        # 非空 treatment 自动覆写对应开关——调用方无需分别传 selector_mode/
        # use_hyperbolic/use_rp1_selector（避免"treatment 只是元数据"）
        if g16_treatment:
            _t = g16_treatment
            if _t == 'random':
                self.selector_mode = 'random'
            elif _t == 'spatial_fixed':
                self.selector_mode = 'spatial_uniform'
            elif _t == 'euc_far':
                self.selector_mode = 'topk'          # largest=True（现状）
            elif _t == 'euc_near':
                self.selector_mode = 'topk'          # near_mode 在 forward 翻方向
            elif _t == 'hyp_far':
                self.selector_mode = 'topk'
                # use_hyperbolic/hyperbolic_type 已在 __init__ 开头映射
                # （P0-1: buffer 注册之前——否则 c 不注册）
            elif _t == 'hyp_near':
                self.selector_mode = 'topk'   # near_mode 在 forward 翻方向（双曲最近）
            elif _t == 'rp1_scorer':
                self.selector_mode = 'topk'
                self.use_rp1_selector = True        # 特征距离+λ×空间新颖性
            elif _t == 'cosine_near':
                # cosine 相似度选点（学长之前 Global Feature-KNN 口径：特征方向相关，
                # 非欧氏距离）。cosine 越大=越相似，largest=True 选最高。
                self.selector_mode = 'topk'
                self.use_cosine = True
            elif _t == 'learned':
                # 可学习远程选点（core idea 1）：无偏候选池 + 可学习 scorer + ST_k 软 top-k。
                # 打分走 candidate−centroid + novelty 标量（Codex P0-2：平移不变 + 保空间信号）。
                self.selector_mode = 'topk'
                self.learned_M = learned_M
                self.learned_delta = learned_delta
                self.learned_scorer = nn.Sequential(
                    nn.Linear(channel + 1, channel),
                    nn.GELU(),
                    nn.Linear(channel, 1))
            elif _t == 'learned_v2':
                # 可学习远程选点 v2：打分器与信号都换，候选池/M/RNG/软 top-k 与 v1 全同
                # （保证 learned vs learned_v2 只差 scorer+signal，可直接归因）。
                # 1) GAT 式 center-candidate 交互（GF-Core 2509.16639 Eq.1）：
                #    e = LeakyReLU(a_c·W_c·centroid + a_n·W_n·candidate)。
                #    注意这是 GAT 的**泛化双线性形式**：原文用共享 W，这里用
                #    独立的 W_c/W_n（center 与 candidate 语义不同，不共享更贴合）。
                #    显式建模两点交互，而非只吃位移差值。
                # 2) 4 路标量信号：cosine 相似度（学长 Global Feature-KNN 有效度量）、
                #    XY 距离、时间跨度、事件密度（POLED/EventFlash：密集=运动=重要）。
                # 3) 位移 MLP 支路保留（能表达 cosine 近邻这类度量），两路相加。
                self.selector_mode = 'topk'
                self.learned_M = learned_M
                self.learned_delta = learned_delta
                self.v2_Wc = nn.Linear(channel, channel, bias=False)
                self.v2_Wn = nn.Linear(channel, channel, bias=False)
                self.v2_a_c = nn.Linear(channel, 1, bias=False)
                self.v2_a_n = nn.Linear(channel, 1, bias=False)
                self.v2_mlp = nn.Sequential(
                    nn.Linear(channel + 4, channel),
                    nn.GELU(),
                    nn.Linear(channel, 1))
            elif _t == 'csnet':
                # CS-Net（2501.10789）per-center 适配：
                #   候选特征 → 偏移注意力（OA: MLP(F−F_sa)+F）→ 贡献分数 → EOT 软 top-k。
                # 与论文的差异：论文是全局选 K 个；我们是每个中心在其候选池里选 G 个，
                # 因此 FE/CA 模块换成本地候选集上的 OA（分块控制 M×M 注意力显存），
                # 打分器吃 [OA 特征; 四路信号]，top-k 用论文的熵正则 OT（非 LapSum）。
                self.selector_mode = 'topk'
                self.learned_M = learned_M
                self.learned_delta = learned_delta
                self.cs_chunk = max(1, int(cs_chunk))  # 每块中心数，控制 [B,s,M,M] 显存
                self.cs_tau = 0.1            # OT 熵正则温度（0.5 太软：有效候选 41 vs k=16）
                self.cs_q = nn.Linear(channel, channel, bias=False)
                self.cs_k = nn.Linear(channel, channel, bias=False)
                self.cs_v = nn.Linear(channel, channel, bias=False)
                self.cs_mlp = nn.Sequential(
                    nn.Linear(channel, channel), nn.GELU(),
                    nn.Linear(channel, channel))
                self.cs_score = nn.Sequential(
                    nn.Linear(channel + 4, channel),
                    nn.GELU(),
                    nn.Linear(channel, 1))
            else:
                raise ValueError(
                    f"g16_treatment 未知: '{_t}'（合法值: random/spatial_fixed/"
                    f"euc_far/euc_near/hyp_far/hyp_near/rp1_scorer/cosine_near/"
                    f"learned/learned_v2/csnet，空串=原行为）")
        if use_hyperbolic:
            # 曲率作为 buffer（选点只需固定几何空间，不参与优化）
            self.register_buffer("c", torch.tensor(curvature_init))

    def _sample_candidates(self, local_idx, B, S, N):
        """无偏候选池：每中心独立采样 M+K 个（排除局部），取前 M 个。

        与 learned/v1 完全一致：有放回采样、同一 RNG、M 由 learned_M 决定。
        """
        M = self.learned_M
        K = local_idx.shape[-1]
        with torch.no_grad():
            cand = torch.randint(0, N, (B, S, M + K), device=local_idx.device,
                                 generator=self._g16_rng_inst)
            is_local = (cand.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1)
            cand = cand.masked_fill(is_local, N)          # 局部点置越界值排最后
            cand_idx = cand.sort(dim=-1).values[..., :M]
            # 极小 N（测试场景）下 K 缓冲可能不够，仍有越界值；钳到 N-1 兜底。
            # 生产配置 N=10240/K=24/M=256 时被掩码的期望数 <1，不会触发。
            return cand_idx.clamp(max=N - 1)

    def _candidate_signals(self, centroid_feat, cand_feat, cand_xyz,
                           center_xyz, all_xyz, cand_idx):
        """learned_v2/csnet 的四路打分信号。

        返回 (cos_sim, d_xy, d_t, density)，均为 [B,S,M]：
          - cosine 相似度（有界，不 z-score）；
          - XY 距离、时间跨度、事件密度：按各中心候选池 z-score，
            只保留相对排序，避免 t 未归一化时量级压过其余三路。
        """
        if center_xyz is None:
            raise ValueError("learned_v2/csnet 需要 center_xyz（[B,S,3] 中心点坐标）")
        center_xyz = center_xyz.unsqueeze(-2)                        # [B,S,1,3]

        def _z(x):
            return (x - x.mean(dim=-1, keepdim=True)) / \
                (x.std(dim=-1, keepdim=True) + 1e-3)

        cos_sim = (F.normalize(centroid_feat, dim=-1).unsqueeze(-2) *
                   F.normalize(cand_feat, dim=-1)).sum(-1)           # [B,S,M]
        d_xy = _z(torch.norm(cand_xyz[..., 1:3] - center_xyz[..., 1:3], dim=-1))
        d_t = _z((cand_xyz[..., 0] - center_xyz[..., 0]).abs())
        dens = index_points(self._voxel_density(all_xyz).unsqueeze(-1),
                            cand_idx).squeeze(-1)                    # [B,S,M]
        return cos_sim, d_xy, d_t, _z(dens)

    def _learned_select(self, centroid_feat, all_feat, local_idx, all_xyz, B, S, C, N, G):
        """learned scorer：无偏候选池 + 可学习 scorer + ST_k 软 top-k（core idea 1）。

        返回 (w, cand_idx)：w [B, S, M] 软权重（Σw≈G，可微），cand_idx [B, S, M] 候选索引。
        训练用软权重加权聚合（梯度回流 scorer），推理硬 top-k。
        Codex 复审（2026-10-24）：M=256、每中心点独立采样、per-epoch 重采（RNG 固定 seed）、
        scorer 输入 candidate−centroid + novelty 标量。
        """
        M = self.learned_M
        K = local_idx.shape[-1]
        _g16_rng = self._g16_rng_inst
        # 1. 无偏候选池（与 v2/csnet 共用同一实现：有放回采样 + 排除 local）
        cand_idx = self._sample_candidates(local_idx, B, S, N)
        # 2. scorer 打分（可微，在 no_grad 外）
        cand_feat = index_points(all_feat, cand_idx)  # [B, S, M, C]
        diff = cand_feat - centroid_feat.unsqueeze(-2)  # candidate−centroid [B,S,M,C]
        cand_xy = index_points(all_xyz, cand_idx)[..., 1:3]  # [B,S,M,2] XY
        local_xy = index_points(all_xyz, local_idx)[..., 1:3]  # [B,S,K,2] XY
        novelty = self._spatial_novelty(cand_xy, local_xy)  # [B,S,M]
        scorer_in = torch.cat([diff, novelty.unsqueeze(-1)], dim=-1)  # [B,S,M,C+1]
        scores = self.learned_scorer(scorer_in).squeeze(-1)  # [B,S,M]
        # 3. LapSum 软 top-k（Σp=k 精确 + 稠密梯度 + 确定性，优于 Gumbel STE/ST_k）
        w = lapsum_soft_topk(scores, G, alpha=0.5)  # [B,S,M]，Σw=k 精确
        return w, cand_idx

    @staticmethod
    def _voxel_density(xyz: torch.Tensor, bins: int = 16) -> torch.Tensor:
        """逐事件密度 = 体素内事件数的 log1p（POLED/EventFlash 启发）。

        事件密集处对应运动/变化剧烈，更值得作为远程点。坐标量纲不限：
        每维按 min-max 归一到 [0,1) 再分格，避免 t 与 x,y 尺度不同导致分格失真。

        Args:
            xyz: [B, N, 3]（t, x, y）
        Returns:
            density: [B, N]，log1p(所在体素事件数)，越密集越大
        """
        with torch.no_grad():
            lo = xyz.amin(dim=1, keepdim=True)                       # [B,1,3]
            hi = xyz.amax(dim=1, keepdim=True)
            span = (hi - lo).clamp(min=1e-6)
            q = ((xyz - lo) / span * bins).floor_().clamp_(0, bins - 1).long()  # [B,N,3]
            key = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]  # [B,N]，bins^3 个体素
            counts = torch.zeros(xyz.shape[0], bins ** 3,
                                 device=xyz.device, dtype=torch.float32)
            counts.scatter_add_(1, key, torch.ones_like(key, dtype=torch.float32))
            return torch.log1p(counts.gather(1, key)).to(xyz.dtype)  # [B,N]

    def _learned_v2_select(self, centroid_feat, all_feat, local_idx, all_xyz,
                           center_xyz, B, S, C, N, G):
        """learned scorer v2：GAT 式交互 + 4 路信号（候选池与 v1 全同）。

        返回 (w, cand_idx)，语义同 _learned_select。

        Args:
            center_xyz: [B, S, 3] 中心点坐标（grouper 层用 fps_idx 取的准确值）。
                learned_v2 下**必填**：缺失时直接报错，不静默退化为局部邻居均值
                （Codex P1：静默兜底会让"中心点取错"这类接线错误无声通过）。
        """
        if center_xyz is None:
            raise ValueError(
                "g16_treatment='learned_v2' 需要 center_xyz（[B,S,3] 中心点坐标）；"
                "LocalRemoteGrouper 会用 fps_idx 传入，直接调用 selector 时须显式提供")
        M = self.learned_M
        K = local_idx.shape[-1]
        # 1. 候选池：与 v1 完全一致（同一 RNG、同一 M、同样排除 local）
        cand_idx = self._sample_candidates(local_idx, B, S, N)

        cand_feat = index_points(all_feat, cand_idx)                 # [B,S,M,C]
        cand_xyz = index_points(all_xyz, cand_idx)                   # [B,S,M,3]
        cos_sim, d_xy, d_t, dens = self._candidate_signals(
            centroid_feat, cand_feat, cand_xyz, center_xyz, all_xyz, cand_idx)

        # 3. GAT 式交互项：e = LeakyReLU(a_c·W_c·centroid + a_n·W_n·candidate)
        #    （a^T[Wh_i‖Wh_j] 的等价分解；LeakyReLU 作用在标量上，与 GAT 一致）
        e = self.v2_a_c(self.v2_Wc(centroid_feat)).unsqueeze(-2) + \
            self.v2_a_n(self.v2_Wn(cand_feat))                       # [B,S,M,1]
        att = F.leaky_relu(e, 0.2).squeeze(-1)                       # [B,S,M]

        # 4. 位移 MLP 支路（保留显式差值，能表达 cosine 近邻这类度量）
        mlp_in = torch.cat([
            cand_feat - centroid_feat.unsqueeze(-2),
            cos_sim.unsqueeze(-1), d_xy.unsqueeze(-1),
            d_t.unsqueeze(-1), dens.unsqueeze(-1)], dim=-1)          # [B,S,M,C+4]
        scores = att + self.v2_mlp(mlp_in).squeeze(-1)               # [B,S,M]

        # 5. 软 top-k：与 v1 同款 LapSum（Σw=k 精确 + 稠密梯度 + 确定性）
        w = lapsum_soft_topk(scores, G, alpha=0.5)
        return w, cand_idx

    def _csnet_select(self, centroid_feat, all_feat, local_idx, all_xyz,
                      center_xyz, B, S, C, N, G):
        """CS-Net（2501.10789）的 per-center 适配：OA + 贡献分数 + EOT 软 top-k。

        与论文的对应关系：
          - FE/CA → 候选集上的偏移注意力 OA(F)=MLP(F−F_sa)+F（F 为候选相对中心特征）；
          - CS  → 贡献分数 MLP（吃 OA 特征 + 四路信号）；
          - Top-k → 熵正则最优传输（eot_soft_topk），不是 LapSum；
          - 训练软加权、推理硬 top-k（由 grouper 的 learned 分支统一处理）。
        与论文的差异：论文全局选 K 个，这里每中心在候选池里选 G 个；
        注意力按中心分块（cs_chunk），避免 [B,S,M,M] 显存爆掉。
        """
        if center_xyz is None:
            raise ValueError("g16_treatment='csnet' 需要 center_xyz（[B,S,3]）")
        cand_idx = self._sample_candidates(local_idx, B, S, N)
        cand_feat = index_points(all_feat, cand_idx)                  # [B,S,M,C]
        cand_xyz = index_points(all_xyz, cand_idx)                    # [B,S,M,3]
        cos_sim, d_xy, d_t, dens = self._candidate_signals(
            centroid_feat, cand_feat, cand_xyz, center_xyz, all_xyz, cand_idx)

        f = cand_feat - centroid_feat.unsqueeze(-2)                   # [B,S,M,C]
        scale = C ** -0.5
        outs = []
        for s0 in range(0, S, self.cs_chunk):
            fc = f[:, s0:s0 + self.cs_chunk]                          # [B,s,M,C]
            q = self.cs_q(fc)
            k = self.cs_k(fc)
            v = self.cs_v(fc)
            att = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
            sa = att @ v
            outs.append(self.cs_mlp(fc - sa) + fc)                    # OA 残差
        f = torch.cat(outs, dim=1)                                    # [B,S,M,C]

        sig = torch.stack([cos_sim, d_xy, d_t, dens], dim=-1)         # [B,S,M,4]
        scores = self.cs_score(torch.cat([f, sig], dim=-1)).squeeze(-1)
        if self.training:
            # 训练：EOT 软权重（梯度回流 scorer）
            w = eot_soft_topk(scores, G, tau=self.cs_tau)             # Σw=G
        else:
            # 推理：硬 top-G（CS-Net 论文口径：训练软、推理硬）
            top = scores.topk(G, dim=-1).indices
            w = torch.zeros_like(scores).scatter_(-1, top, 1.0)
        return w, cand_idx

    def _cosine_sim(self, centroid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """cosine 相似度：centroid [B,S,C] × points [B,c,C] → [B,S,c]（越大越相似）。"""
        c = F.normalize(centroid, dim=-1)          # [B,S,C]
        p = F.normalize(points, dim=-1)            # [B,c,C]
        return torch.bmm(c, p.transpose(-1, -2))   # [B,S,c]

    def _hyp_dist_chunk(self, centroid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """分块双曲距离计算。

        Args:
            centroid: [B, S, C] 中心点欧氏特征
            points: [B, c, C] 当前块的点欧氏特征
        Returns:
            dist: [B, S, c] 双曲距离
        """
        c = torch.abs(self.c) + 1e-4
        if self.hyperbolic_type == "poincare":
            centroid_h = poincare_exp_origin(centroid, c)  # [B, S, C]
            points_h = poincare_exp_origin(points, c)      # [B, c, C]
            return poincare_distance(
                centroid_h.unsqueeze(2), points_h.unsqueeze(1), c)
        else:  # lorentz
            centroid_h = lorentz_exp_origin(centroid, c)  # [B, S, C+1]
            points_h = lorentz_exp_origin(points, c)      # [B, c, C+1]
            return lorentz_distance(
                centroid_h.unsqueeze(2), points_h.unsqueeze(1), c)

    @staticmethod
    def _exclude_local(chunk_dist: torch.Tensor, local_idx: torch.Tensor,
                       start: int, sentinel: float = float("-inf")) -> torch.Tensor:
        """把落在本 chunk [start, start+c) 内的局部邻居距离置为 sentinel。

        Args:
            chunk_dist: [B, S, c] 当前块的距离矩阵
            local_idx: [B, S, K] 局部邻居的全局索引
            start: 当前块在 N 维上的起始偏移
            sentinel: 排除哨兵值。topk largest=True 用 -inf（默认）；
                largest=False（euc_near 方向翻转）必须用 +inf——
                否则 -inf 作为"最小值"会被 near 方向的 topk 优先选中。
        """
        B, S, c = chunk_dist.shape
        offset = local_idx - start  # [B, S, K]
        valid = (offset >= 0) & (offset < c)
        # 无效邻居统一散射到第 c 列（末尾补的一列 sentinel），避免越界
        safe_idx = torch.where(valid, offset, torch.full_like(offset, c))
        padded = torch.cat(
            [chunk_dist, chunk_dist.new_full((B, S, 1), sentinel)], dim=-1)
        padded.scatter_(2, safe_idx, sentinel)
        return padded[..., :c]

    @staticmethod
    def _spatial_novelty(cand_xy: torch.Tensor, local_xy: torch.Tensor) -> torch.Tensor:
        """RP1 空间新颖性：候选点与局部邻居集合的最小欧氏距离。

        空间新颖性大 = 候选点在空间上离已覆盖的局部邻域远（位置新颖），
        与特征距离互补：既"特征不相似"又"空间位置新颖"的点才是好的远程点。

        坐标维度由调用方决定（Codex 第四轮记录#1）：
          - g16_treatment='rp1_scorer'：二维 XY（RP1 原文 Spatial novelty =
            XY distance，与 profile agent 的 nov 口径一致）
          - 旧 --use_rp1_selector（空 treatment）：三维 (t,x,y) 原口径
            ——旧 checkpoint 推理行为不变

        Args:
            cand_xy: [B, S, M, D] 或 [B, 1, M, D] 候选点空间坐标（D=2 或 3）
                （chunk 路径下 S 维为 1，靠广播对齐每个中心点的邻居）
            local_xy: [B, S, K, D] 各中心点局部邻居的空间坐标
        Returns:
            novelty: [B, S, M] 每个候选点到该中心点局部邻居的最小欧氏距离

        显存优化：逐个局部邻居计算 [B, S, M] 距离并滚动取 min，
        避免一次性物化 [B, S, M, K] 的完整距离张量（与分块 top-k 同思路）。
        """
        K = local_xy.shape[2]
        min_sq = None
        for k in range(K):
            # [B, S, M, D] 广播相减后按坐标维求平方和 → [B, S, M]
            d = ((cand_xy - local_xy[:, :, k:k + 1, :]) ** 2).sum(-1)
            min_sq = d if min_sq is None else torch.minimum(min_sq, d)
        # 开方得到欧氏距离（加 eps 防 0 距离时梯度/数值问题）
        return torch.sqrt(min_sq + 1e-12)

    @staticmethod
    def _greedy_div_select(cand_dist: torch.Tensor, cand_points: torch.Tensor,
                           G: int, div_lambda: float) -> torch.Tensor:
        """贪心 max-min 多样性选择（在候选并集上，替代独立 top-G 打分）。

        动机：独立 top-k 打分会让 G 个远程点全落在同一特征方向（冗余）。
        贪心 max-min：第 1 个点取得分最高的；后续每个点选
            score = 得分 + div_lambda * (与已选集合的最小特征距离)
        最大的，既保证高分又保证特征方向彼此分散。

        Args:
            cand_dist: [B, S, M] 候选点得分（含 -inf 排除项）
            cand_points: [B, S, M, C] 候选点特征
            G: 需要选取的点数
            div_lambda: 多样性项权重
        Returns:
            final: [B, S, G] 选中的候选局部索引（保持贪心选取顺序）
        """
        # 第 1 个点：得分最高的（-inf 排除项不会被选中）
        sel = cand_dist.argmax(dim=-1, keepdim=True)  # [B, S, 1]
        for _ in range(G - 1):
            g = sel.shape[-1]
            # 已选点特征 [B, S, g, C]
            sel_pts = torch.gather(
                cand_points, 2,
                sel.unsqueeze(-1).expand(-1, -1, -1, cand_points.shape[-1]))
            # 每个候选点到已选集合的最小特征距离 [B, S, M]
            d = ((cand_points.unsqueeze(3) - sel_pts.unsqueeze(2)) ** 2).sum(-1)
            d_sel = d.min(dim=-1)[0]
            score = cand_dist + div_lambda * d_sel
            # 已选点置 -inf，不重复选
            score = score.scatter(2, sel, float("-inf"))
            sel = torch.cat(
                [sel, score.argmax(dim=-1, keepdim=True)], dim=-1)  # [B, S, g+1]
        # 用 mask 把贪心选中序转回 top-k 索引形式（保持选取顺序，非按分值排序）
        mask = torch.zeros_like(cand_dist)
        mask.scatter_(2, sel, 1.0)
        return (cand_dist * mask - 1e9 * (1 - mask)).topk(
            G, dim=-1, largest=True)[1]

    @staticmethod
    def _spatial_uniform_select(cand_idx: torch.Tensor, all_xyz: torch.Tensor,
                                G: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """spatial_uniform：原始 x,y 空间均匀网格采样（RP1 论文最好的 cheap selector）。

        把 x,y ∈ [0,1] 均分为 grid_size × grid_size 网格（grid_size =
        ceil(√G)，如 G=16 → 4×4=16 格正好每格 1 点），每个网格内从候选
        并集中随机选 1 个点（如果该格有候选），按随机网格访问序依次收集
        直到凑够 G 个；非空网格不足 G 个时用随机候选补齐。

        与 random 模式一样不打分、不计算 cand_dist，直接在候选并集
        （分块阶段已排除局部邻居）上操作，避免物化 [B, S, N] 大张量；
        候选并集本身逐中心点不同（块内随机采样逐 (b,s) 独立），因此
        各中心点的网格采样结果自然不同。

        Args:
            cand_idx: [B, S, M] 候选并集索引
            all_xyz: [B, N, 3] 全部点的空间坐标（x, y, t）
            G: 需要选取的点数
        Returns:
            remote_idx: [B, S, G] 选中的远程邻居索引（未排序，调用方排序）
        """
        B, S, M = cand_idx.shape
        grid_size = math.ceil(math.sqrt(G))
        n_cells = grid_size * grid_size
        cand_xyz = index_points(all_xyz, cand_idx)  # [B, S, M, 3]
        # 网格编号 cell_x * grid_size + cell_y ∈ [0, n_cells)。
        # 坐标应已归一化到 [0,1]，clamp 兜底防乘 grid_size 后越界
        eps = 1e-6
        # 2026-10-15 列错位修正（Codex 审查 #8）：all_xyz 布局是 (t, x, y)，
        # 原版 [...,0]/[...,1] 实际按 (t,x) 分格——y 被忽略。修正为 x=[...,1],
        # y=[...,2]。注: g16_treatment="spatial_fixed" 时生效的正是本方法。
        # 兼容性声明（Codex 第三轮补充#5）: 本修正**无条件**作用于旧
        # selector_mode="spatial_uniform" 路径（空 treatment 亦生效）——
        # 旧实验若需复现"按 (t,x) 分格"的历史行为，无开关可回退；所有
        # spatial_uniform 口径的旧结果应视为列错位口径、不复作为对照。
        # 决策: 修正值是正确几何语义（y 参与分格），选择无条件生效并
        # 显式记录此兼容性变化，而非为错误行为保留开关。
        cell_x = (cand_xyz[..., 1].clamp(0.0, 1.0 - eps) * grid_size).long()
        cell_y = (cand_xyz[..., 2].clamp(0.0, 1.0 - eps) * grid_size).long()
        cell = cell_x * grid_size + cell_y              # [B, S, M]
        rand = torch.rand(B, S, M, device=cand_idx.device, generator=generator)  # 每个候选的随机分数
        # 把"逐格收集 + 不足补齐"编码为单次 topk 的优先级分数：
        #   网格赢家 ∈ (1, 2]，网格访问序越早优先级越高（2 - seq/n_cells）
        #   其余候选 = rand ∈ [0, 1)，作非空网格不足 G 时的随机补齐池
        priority = rand.clone()
        # 网格访问顺序随机化（"直到凑够 G 个"时先访问哪些格是随机的）
        cell_order = torch.randperm(n_cells, device=cand_idx.device, generator=generator)
        for seq, j in enumerate(cell_order.tolist()):
            in_cell = (cell == j)                        # [B, S, M]
            if not in_cell.any():
                continue  # 该格全 batch 无候选，跳过
            # 格内随机分数最大的候选为赢家（该格为空的 (b,s) 处 argmax
            # 无意义，下面用 has 屏蔽不写入）
            winner = rand.masked_fill(~in_cell, float("-inf")).argmax(-1)  # [B, S]
            has = in_cell.any(-1)                        # [B, S] 该中心点在此格是否有候选
            win_priority = 2.0 - seq / n_cells           # ∈ (1, 2]，seq 越小越优先
            old = priority.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
            priority.scatter_(
                -1, winner.unsqueeze(-1),
                torch.where(has, torch.full_like(old, win_priority), old).unsqueeze(-1))
        # 先取各网格赢家（按随机格序），非空网格不足 G 时自动用随机候选补齐
        final = priority.topk(G, dim=-1)[1]              # [B, S, G]
        return torch.gather(cand_idx, -1, final)

    def forward(self, centroid_feat: torch.Tensor, all_feat: torch.Tensor,
                local_idx: torch.Tensor,
                global_anchor: torch.Tensor | None = None,
                all_xyz: torch.Tensor | None = None,
                center_xyz: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            centroid_feat: [B, S, C] 中心点特征
            all_feat: [B, N, C] 全部点特征
            local_idx: [B, S, K] 局部邻居索引（待排除）
            global_anchor: [B, S, C] 或 None。全局吸收节点特征（PVAG 启发）；
                非 None 时混合距离 = 到中心点距离 + anchor_lambda * 到锚点距离。
            all_xyz: [B, N, 3] 或 None。全部点的空间坐标（t, x, y），
                仅 use_rp1_selector=True 时用于计算空间新颖性（RP1），
                或 selector_mode="spatial_uniform" 时用于 x,y 网格采样。
            center_xyz: [B, S, 3] 或 None。中心点自身坐标，仅 learned_v2 使用
                （时间跨度/XY 距离信号）；learned_v2 下必填，缺失直接报错，
                其余 treatment 忽略此参数。
        Returns:
            remote_idx: [B, S, G] 远程邻居索引（沿 G 维升序）
        """
        N = all_feat.shape[1]
        B, S, C = centroid_feat.shape
        G = self.remote_g
        # ==== G16 分派激活（空 treatment 时全部跳过,原行为不动）====
        g16_on = bool(getattr(self, 'g16_treatment', ''))
        # euc_near: 方向翻转一行（原文 KNN 语义,Codex 清单#5）
        # near_mode 只含 euc_near/hyp_near（距离语义：near=largest=False 选最小）。
        # 注意：cosine_near 语义也是"near"（选最相似），但 cosine 相似度越大越相似，
        # 方向是 largest=True，与 euc/hyp 相反——所以 cosine_near 必须保持不在此元组内，
        # 否则 largest 翻反会静默选"最不相似"（Codex P2：防误加）。
        near_mode = g16_on and self.g16_treatment in ('euc_near', 'hyp_near')
        # Codex 第三轮 P0-2 修复：惰性创建一次、持续复用同一 generator——
        # 不再每次 forward 新建 + manual_seed（那会把随机基线变成逐调用
        # 固定，同输入两次调用结果相同，统计语义被破坏）。实例级属性
        # 首调时按 seed=rng_seed*1000+stage_idx 建立后不再重播种。
        # device 跟随首调输入（CPU 训练/HPC CUDA）。
        if g16_on:
            _g16_rng = getattr(self, '_g16_rng_inst', None)
            if _g16_rng is None:
                _g16_rng = torch.Generator(device=all_feat.device)
                _g16_rng.manual_seed(self._g16_rng_seed)
                self._g16_rng_inst = _g16_rng
            elif _g16_rng.device != all_feat.device:
                # 设备迁移（理论场景）：重建并重播种（保持确定性可复现）
                _g16_rng = torch.Generator(device=all_feat.device)
                _g16_rng.manual_seed(self._g16_rng_seed)
                self._g16_rng_inst = _g16_rng
            if all_xyz is None:
                raise ValueError(
                    f"g16_treatment='{self.g16_treatment}' 需要 forward 传入 "
                    f"all_xyz（[B,N,3] (t,x,y) 空间坐标,供空间统计与分格）")
        if N < G + local_idx.shape[-1]:
            raise ValueError(
                f"N={N} 不足以同时容纳 K={local_idx.shape[-1]} 局部邻居与 G={G} 远程邻居")
        if self.use_rp1_selector and all_xyz is None:
            raise ValueError("use_rp1_selector=True 时 forward 需要传入 all_xyz（[B, N, 3] 空间坐标）")
        if self.selector_mode == "spatial_uniform" and all_xyz is None:
            raise ValueError(
                'selector_mode="spatial_uniform" 时 forward 需要传入 all_xyz（[B, N, 3] 空间坐标）')
        # RP1：预取各中心点局部邻居的空间坐标，供空间新颖性计算复用。
        # Codex 第四轮记录#1: 二维 XY（rp1_scorer treatment 口径，RP1 原文
        # Spatial novelty = XY distance）只在 g16_treatment='rp1_scorer' 时
        # 生效；旧 --use_rp1_selector 路径（空 treatment）保持三维 (t,x,y)
        # ——旧 checkpoint 在新代码下推理口径不变，兼容性无破坏。
        _rp1_2d = bool(getattr(self, 'g16_treatment', '')) and \
            self.g16_treatment == 'rp1_scorer'
        if self.use_rp1_selector:
            _xyz_cols = slice(1, 3) if _rp1_2d else slice(0, 3)
            local_nxyz = index_points(all_xyz, local_idx)[..., _xyz_cols]
        # 全局锚点扩展为 [B, S, C]（可学习单向量 -> 广播到每个中心点）
        if global_anchor is not None:
            if global_anchor.dim() == 1:
                global_anchor = global_anchor.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        # ==== learned scorer 分支（core idea 1）：无偏池 + scorer + ST_k 软 top-k ====
        if g16_on and self.g16_treatment in ('learned', 'learned_v2', 'csnet'):
            if self.g16_treatment == 'csnet':
                return self._csnet_select(
                    centroid_feat, all_feat, local_idx, all_xyz,
                    center_xyz, B, S, C, N, G)
            if self.g16_treatment == 'learned_v2':
                return self._learned_v2_select(
                    centroid_feat, all_feat, local_idx, all_xyz,
                    center_xyz, B, S, C, N, G)
            return self._learned_select(centroid_feat, all_feat, local_idx, all_xyz,
                                        B, S, C, N, G)
        # 选择只产生离散索引，无需梯度；no_grad 避免距离矩阵挂计算图浪费显存
        with torch.no_grad():
            topk_list = []
            for start in range(0, N, self.chunk):
                end = min(start + self.chunk, N)
                c = end - start
                # 尾块可能不足 G 个元素：块内取 min(G, c) 个，不足则整块全作候选
                k = min(G, c)
                if self.selector_mode in ("random", "spatial_uniform"):
                    # random / spatial_uniform（RP1 cheap selector）：不打分，
                    # 块内直接随机采样 k 个非局部候选（随机打分 top-k 等价
                    # 无放回均匀采样，_exclude_local 把本块内局部邻居置
                    # -inf），省掉块内特征距离计算；候选并集即为逐 (b,s)
                    # 独立的近似均匀随机子集（每点入选概率 ≈ G/(N-K)）
                    if _g16_rng is not None:
                        # G16 RNG 隔离: 独立 generator（不消耗训练全局 RNG）
                        chunk_rand = torch.rand(B, S, c, device=all_feat.device,
                                                generator=_g16_rng)
                    else:
                        chunk_rand = torch.rand(B, S, c, device=all_feat.device)
                    chunk_rand = self._exclude_local(chunk_rand, local_idx, start)
                    topk_list.append(chunk_rand.topk(k, dim=-1)[1] + start)
                    continue
                if self.use_cosine:
                    chunk_dist = self._cosine_sim(centroid_feat, all_feat[:, start:end, :])
                elif self.use_hyperbolic:
                    chunk_dist = self._hyp_dist_chunk(
                        centroid_feat, all_feat[:, start:end, :])
                else:
                    chunk_dist = square_distance(centroid_feat, all_feat[:, start:end, :])
                if self.use_rp1_selector:
                    # RP1 空间新颖性（Codex 第三轮补充#4+第四轮记录#1: 二维 XY
                    # 仅限 rp1_scorer treatment——RP1 原文 Spatial novelty =
                    # XY distance；旧 --use_rp1_selector 路径保持三维 (t,x,y)
                    # 口径不变）。块内候选 [B,1,c,C']（广播）与局部邻居的
                    # 最小欧氏距离 [B,S,c]
                    chunk_novelty = self._spatial_novelty(
                        all_xyz[:, start:end, _xyz_cols].unsqueeze(1), local_nxyz)  # [B, S, c]
                    # RP1 原文语义：好远程点 = 特征相关（最近）+ 空间远程（新颖）。
                    # feature 项取负（距离小=相似=得分高），novelty 保持正（远=好）。
                    # 方向修复 2026-10-23：原 chunk_dist 取正=选"最不相似"，与
                    # 原文 Global Feature-KNN（特征最近）相反，已翻负。
                    chunk_dist = -chunk_dist + self.rp1_lambda * chunk_novelty
                if global_anchor is not None:
                    # PVAG 式全局吸收：混合"到中心点"与"到全局锚点"的距离
                    if self.use_hyperbolic:
                        anchor_dist = self._hyp_dist_chunk(
                            global_anchor, all_feat[:, start:end, :])
                    else:
                        anchor_dist = square_distance(global_anchor, all_feat[:, start:end, :])
                    chunk_dist = chunk_dist + self.anchor_lambda * anchor_dist
                # g16 euc_near: 方向翻转时排除哨兵也翻（-inf 在 largest=False
                # 的 topk 里是"最小值"会被优先选中 → 局部邻居全被选中!）
                # near 模式排除用 +inf（"最大值"→ largest=False 的 topk 不选它）
                _blk_sentinel = float('inf') if near_mode else float('-inf')
                chunk_dist = self._exclude_local(
                    chunk_dist, local_idx, start, sentinel=_blk_sentinel)
                _blk_largest = not near_mode
                topk_list.append(chunk_dist.topk(k, dim=-1, largest=_blk_largest)[1] + start)
            cand_idx = torch.cat(topk_list, dim=-1)  # [B, S, n_chunks*G]
            if self.selector_mode == "random":
                # random（RP1 随机基线）：候选并集（分块阶段已随机收集并
                # 排除局部邻居）中随机选 G 个，不打分、不算 cand_dist。
                # 极端情形（尾块 c<G+K）下 -inf 排除项可能被块内 topk 带
                # 入 cand_idx，成员检测剔除，保证远程点与局部邻居不重叠
                M = cand_idx.shape[-1]
                if _g16_rng is not None:
                    rand = torch.rand(B, S, M, device=cand_idx.device,
                                      generator=_g16_rng)
                else:
                    rand = torch.rand(B, S, M, device=cand_idx.device)
                is_local = (cand_idx.unsqueeze(-1)
                            == local_idx.unsqueeze(-2)).any(-1)      # [B, S, M]
                rand = rand.masked_fill(is_local, float("-inf"))
                final = rand.topk(G, dim=-1)[1]                      # [B, S, G]
                remote_idx = torch.gather(cand_idx, -1, final)
                self._g16_cache(remote_idx, local_idx, all_xyz, all_feat)
                return remote_idx.sort(dim=-1)[0]
            if self.selector_mode == "spatial_uniform":
                # spatial_uniform（RP1 最好的 cheap selector）：x,y 空间
                # 均匀网格采样，同样不打分、不算 cand_dist
                _ri = self._spatial_uniform_select(
                    cand_idx, all_xyz, G, generator=_g16_rng)
                self._g16_cache(_ri, local_idx, all_xyz, all_feat)
                return _ri.sort(dim=-1)[0]
            # 候选并集上重算距离，取全局 top-G（topk / greedy_div 原有逻辑）
            cand_points = index_points(all_feat, cand_idx)  # [B, S, n_chunks*G, C]
            if self.use_cosine:
                # cosine 相似度：cand_points [B,S,M,C] × centroid [B,S,1,C] → [B,S,M]
                c = F.normalize(centroid_feat, dim=-1).unsqueeze(-2)   # [B,S,1,C]
                p = F.normalize(cand_points, dim=-1)                  # [B,S,M,C]
                cand_dist = (c * p).sum(-1)                           # [B,S,M]
            elif self.use_hyperbolic:
                # 双曲距离：按 batch*group 展平后批量计算，再恢复形状
                Bc, Sc, Nc, Cc = cand_points.shape
                centroid_flat = centroid_feat.reshape(Bc * Sc, 1, Cc)
                cand_flat = cand_points.reshape(Bc * Sc, Nc, Cc)
                cand_dist = self._hyp_dist_chunk(
                    centroid_flat, cand_flat
                ).reshape(Bc, Sc, Nc)
            else:
                cand_dist = ((cand_points - centroid_feat.unsqueeze(-2)) ** 2).sum(-1)
            if self.use_rp1_selector:
                # RP1：候选并集上同样加上空间新颖性，保持与块内一致的得分定义
                # （Codex 第三轮补充#4+第四轮记录#1: 二维 XY 仅限 rp1_scorer）
                cand_nxyz = index_points(all_xyz, cand_idx)[..., _xyz_cols]  # [B,S,M,2/3]
                # 与块内一致：feature 项取负（特征最近=相关），novelty 保持正。
                cand_dist = -cand_dist + self.rp1_lambda * self._spatial_novelty(
                    cand_nxyz, local_nxyz)
            if global_anchor is not None:
                if self.use_hyperbolic:
                    Bc, Sc, Nc, Cc = cand_points.shape
                    anchor_flat = global_anchor.reshape(Bc * Sc, 1, Cc)
                    cand_flat = cand_points.reshape(Bc * Sc, Nc, Cc)
                    cand_anchor_dist = self._hyp_dist_chunk(
                        anchor_flat, cand_flat
                    ).reshape(Bc, Sc, Nc)
                else:
                    cand_anchor_dist = (
                        (cand_points - global_anchor.unsqueeze(-2)) ** 2).sum(-1)
                cand_dist = cand_dist + self.anchor_lambda * cand_anchor_dist
            if self.selector_mode == "greedy_div" and self.div_lambda != 0.0:
                # 贪心 max-min 多样性选择：score = 得分 + λ * 到已选集合的最小
                # 特征距离，避免 G 个远程点全落在同一特征方向。G 次循环、
                # 候选数 M≈n_chunks*G（~160），开销可忽略
                final = self._greedy_div_select(
                    cand_dist, cand_points, G, self.div_lambda)
            else:
                # 原版独立 top-G 打分（topk 模式，或 div_lambda=0 完全回退原行为）
                # g16 euc_near: 并集层同样翻方向（与块内一致）
                final = cand_dist.topk(G, dim=-1, largest=not near_mode)[1]
            remote_idx = torch.gather(cand_idx, -1, final)
        self._g16_cache(remote_idx, local_idx, all_xyz, all_feat)
        return remote_idx.sort(dim=-1)[0]

    def _g16_cache(self, remote_idx, local_idx, all_xyz, all_feat=None):
        """G16 选点缓存生产（Codex 清单#2 闭环的生产侧）——统一出口。

        g16_treatment 非空且 _g16_sink 已挂载时，把当次选点记入 sink
        （LocalRemoteGrouper.forward 挂到 net._g16_stage_selections 链）。
        所有 return 路径（topk/random/spatial_uniform/near）都在返回前
        调用本方法——early return 不可达问题已修（v2 勘误）。
        Codex 第三轮补充#3: 保存 feat（[B,N,C] 点特征，detached）——
        trainer 的 sel_rec.get('feat') 用于 profile agent 的 feat_cos_*，
        不传则该三项永远为 None。
        """
        if not getattr(self, 'g16_treatment', ''):
            return
        sink = getattr(self, '_g16_sink', None)
        if sink is None:
            return
        sink.append({
            'stage': getattr(self, 'stage_idx', 0),
            'remote': remote_idx.detach(),
            'local': local_idx.detach(),
            'centroid': None,   # 由 LocalRemoteGrouper 层补填（fps_idx 在那）
            'xytp': all_xyz.detach() if all_xyz is not None else None,
            'feat': all_feat.detach() if all_feat is not None else None,
        })


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

    数值稳定性（v3.2，修复反向梯度爆炸 NaN）：
      - key L2 归一化：保证 (I - β k kᵀ) 为收缩映射，BPTT 梯度有界
        （未归一化时 24 步反传按 |1-β‖k‖²|^K 指数爆炸）
      - α 先 clamp logit 再 exp：避免 exp 溢出 inf 后 clamp 反传 0*inf=NaN
      - 状态范数截断 detach 且只降不升：梯度不走 1/norm² 路径
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
            # 状态范数截断（防前向漂移）。v3.2 修复：
            #   - norm 下界 1.0：只降尺度不放大（旧版 min=1e-6 会产生 ~5e7 缩放系数）
            #   - norm detach：范数只是数值稳定器，梯度不应流经 1/norm² 路径
            #     （旧版 norm 参与反传，配合极大状态有 inf*0 → NaN 风险）
            state_norm = state.norm(dim=(-2, -1), keepdim=True).clamp(min=1.0).detach()
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
        # v3.2 修复（反向梯度爆炸根因）：key L2 归一化（DeltaNet/KDA 标准做法）。
        # 递推对状态的雅可比含 (I - β k kᵀ)，其沿 k 的特征值为 1-β‖k‖²。
        # 未归一化时 ‖k‖²>2/β 即模长>1，K=24 步 BPTT 梯度按 |1-β‖k‖²|^K
        # 指数放大直至 inf→NaN；归一化后特征值 ∈ [1-β, 1]，收缩映射，梯度有界。
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta_proj(x_sorted)).clamp(max=0.5)
        # v3.2 修复：先 clamp logit 再 exp。旧版 exp→inf 后 clamp 反传 0*inf=NaN；
        # clamp 到 (-∞, 0] 保证 exp ∈ (0, 1]，下界 log(0.05) 等价旧版 min=0.05
        alpha = torch.exp(
            self.alpha_proj(x_sorted).clamp(min=math.log(0.05), max=0.0))
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
        # Lorentz 径向约束正则（lorentz 分支每次 _aggregate 后更新，供训练循环加到 loss）
        self.radial_radius = 4.0  # R_t
        self.last_radial_pen = None

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
            # Lorentz 径向约束：限制切空间表示的范数不超过 R_t，
            # 超出部分二次惩罚（hinge^2），防止切向量漂移过远导致数值不稳
            R_t = self.radial_radius
            pen_x = F.relu(x_tangent.norm(dim=-1) - R_t) ** 2  # [M, K]
            pen_agg = F.relu(agg_tangent.norm(dim=-1) - R_t) ** 2  # [M]
            self.last_radial_pen = (pen_x.mean() + pen_agg.mean()) / 2
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
                 remote_kda: bool = False,
                 hyperbolic_type: str = "poincare",
                 use_hyperbolic_selector: bool = False,
                 remote_dropout: float = 0.0,
                 remote_stochastic_depth: float = 0.1,
                 use_rp1_selector: bool = False,
                 rp1_lambda: float = 1.0,
                 selector_mode: str = "topk",
                 div_lambda: float = 1.0,
                 remote_gate: bool = False,
                 # G16 六处理分派（2026-10-15）
                 g16_treatment: str = "",
                 g16_rng_seed: int = 42,
                 stage_idx: int = 0,
                 learned_M: int = 256,
                 cs_chunk: int = 256,
                 **kwargs):
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
            remote_kda: 是否对远程 G 个点做 KDA 因果时间编码（EventMamba
                启发）。远程分支原本是纯集合操作（无序双曲注意力聚合），
                远程点之间的时序关系丢失；启用时复用 KDACausalEncoder
                对远程点按时间戳做 gated delta rule 扫描，残差加法同局部
                KDA 的接法。默认 False 保证旧实验可复现。
            hyperbolic_type: "poincare"（庞加莱球，原版）或 "lorentz"
                （洛伦兹模型，v3.1 新增，消除边界梯度尖峰）
            remote_dropout: 远程聚合输出后的 dropout 概率（缓解双曲过拟合）
            remote_stochastic_depth: 远程聚合支路 stochastic depth 概率
                p_drop（训练时逐样本以该概率跳过双曲远程聚合，直接用组均值
                残差代替；推理时不跳过。0 关闭）
            use_rp1_selector: RP1 空间新颖性远程选择器（消融实验）。
                True 时远程选择得分 = 特征距离 + rp1_lambda * 空间新颖性
                （候选点与局部邻居的最小空间欧氏距离，RP1 论文：结合
                spatial novelty 的选择器优于纯特征距离 top-k）。
                False 时行为与原版完全一致。
            rp1_lambda: 空间新颖性权重（仅 use_rp1_selector=True 时有效）
            selector_mode: 远程点选择策略 "topk"（原版）/ "greedy_div"
                （贪心 max-min 多样性选择）/ "random"（候选并集随机采样，
                RP1 论文随机基线）/ "spatial_uniform"（x,y 均匀网格采样，
                RP1 论文最好的 cheap selector），详见 RemoteSelector
            div_lambda: 多样性项权重（仅 selector_mode="greedy_div" 时有效）
            remote_gate: 是否启用远程分支门控融合（SECNet 启发的 gated fusion）。
                True 时 gate = sigmoid(MLP([remote_feat, centroid_feat]))，
                remote_feat = remote_feat * gate，由网络学习逐通道抑制无效的
                远程特征；最后一层初始化 weight=0、bias=2.0，起步 gate≈0.88
                ≈直连，与未启用时行为接近。False 时保持原直连行为不变
                （保证旧实验可复现）。
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
        # 远程 KDA 因果时间编码（可选，EventMamba 启发）：远程 G 点按时间戳
        # 扫描，恢复纯集合操作丢失的远程点间时序关系；v3.2 数值修复自动继承
        self.remote_kda_enc = KDACausalEncoder(channel) if remote_kda else None
        # Remote 分支：远程邻居选择 + 双曲聚合
        self.remote_selector = RemoteSelector(remote_g=remote_g,
                                              # 双曲距离矩阵是 [B,S,N,C+1] 向量（欧氏是标量），显存贵 ~65 倍。
                                              # 分块（chunk）决定两个矩阵：块内 ∝ chunk、候选并集 ∝ N/chunk*G，
                                              # 平衡点 chunk≈sqrt(N*G)=sqrt(10240*16)≈405，取 512。
                                              # batch16 下无论 chunk 怎么调都 ~10GB 会 OOM，须配合降 batch（8）。
                                              chunk=512,
                                              anchor_lambda=anchor_lambda,
                                              use_hyperbolic=use_hyperbolic_selector,
                                              curvature_init=curvature_init,
                                              hyperbolic_type=hyperbolic_type,
                                              use_rp1_selector=use_rp1_selector,
                                              rp1_lambda=rp1_lambda,
                                              # G16 六处理分派透传（2026-10-15）
                                              g16_treatment=g16_treatment,
                                              stage_idx=stage_idx,
                                              rng_seed=g16_rng_seed,
                                              selector_mode=selector_mode,
                                              div_lambda=div_lambda,
                                              learned_M=learned_M,
                                              cs_chunk=cs_chunk,
                                              channel=channel)
        if hyperbolic_type == "euclidean":
            self.hyper_agg = EuclideanRemoteAggregator(channel, temperature_init=1.0)
        else:
            self.hyper_agg = HyperbolicRemoteAggregator(channel, curvature_init=curvature_init,
                                                         hyperbolic_type=hyperbolic_type)
        # ==== 逐 stage 集成断言（Codex G16 hyp_far 复审 P0-1）====
        # 防止"selector 走 lorentz 但 aggregator 回落 poincare"的分裂态。
        # 仅 hyp_far/hyp_near 强校验（六处理中走双曲距离选点的链）；空串=历史 topk 行为不拦。
        if g16_treatment in ('hyp_far', 'hyp_near'):
            assert self.remote_selector.use_hyperbolic, \
                f"[stage {stage_idx}] hyp_far 但 selector.use_hyperbolic=False"
            assert self.remote_selector.hyperbolic_type == 'lorentz', \
                f"[stage {stage_idx}] hyp_far 但 selector.hyperbolic_type={self.remote_selector.hyperbolic_type}"
            assert getattr(self.hyper_agg, 'hyperbolic_type', None) == 'lorentz', \
                f"[stage {stage_idx}] hyp_far 但 aggregator.hyperbolic_type={getattr(self.hyper_agg, 'hyperbolic_type', None)}"
        # ==== P0-3 修复（Codex）: sink 链路接通 ====
        # g16_treatment 非空时，本 grouper 持有一个 sink 容器并挂到
        # remote_selector。LRHyperNet 的 stage loop 会把所有 grouper 的
        # sink 汇聚到 net._g16_stage_selections（见 lr_hypernet 补丁）。
        self._g16_stage_sink = [] if g16_treatment else None
        if g16_treatment:
            self.remote_selector._g16_sink = self._g16_stage_sink
        # 全局吸收节点（PVAG 启发）：每 stage 一个可学习锚点向量
        self.use_global_anchor = use_global_anchor
        if use_global_anchor:
            self.global_anchor = nn.Parameter(torch.zeros(channel))
        # 远程聚合输出 dropout（缓解双曲过拟合）
        self.remote_dropout = nn.Dropout(remote_dropout)
        # 远程聚合支路 stochastic depth 概率（训练时生效）
        self.remote_stochastic_depth = remote_stochastic_depth
        # 远程分支门控融合（SECNet 启发）：由 [remote_feat, centroid_feat]
        # 生成逐通道门控，学习性抑制无效远程特征；默认关闭保持原行为
        self.use_remote_gate = remote_gate
        if remote_gate:
            # 门控 MLP：Linear(2C→C) → GELU → Linear(C→C)
            self.remote_gate = nn.Sequential(
                nn.Linear(channel * 2, channel),
                nn.GELU(),
                nn.Linear(channel, channel),
            )
            # 初始化：最后一层 weight=0、bias=2.0，起步 gate=sigmoid(2)≈0.88，
            # 近似全通（≈原直连行为），训练中由梯度自行学习开/关门
            nn.init.zeros_(self.remote_gate[-1].weight)
            nn.init.constant_(self.remote_gate[-1].bias, 2.0)

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
        # xytp[..., 0:3]（x, y, t 空间坐标）传给 RemoteSelector，
        # 仅 use_rp1_selector=True 时用于计算空间新颖性；否则被忽略，行为不变
        # ==== G16 分派挂载（2026-10-15）: grouper 层把 sink 容器传给 selector ====
        # net._g16_stage_selections 由 trainer 侧创建（--g16_profile 时）；
        # 此处把本 stage 的收集容器挂到 selector 上,选点后补填 centroid。
        # Codex 第三轮 P0-1 修复：本地 sink 是**独立列表**（不指向 net 全局
        # 列表）；清空移到 forward **开头**（清上次 forward 的残留）而非末尾
        # ——末尾清会让记录在 LRHyperNet stage 后 extend 之前就消失。
        # 数据流：selector append → 本地 sink → LRHyperNet stage 后 extend
        # 到 net._g16_stage_selections 并清本地（profiling off 时残留由
        # 下次 forward 开头的 clear 兜底——内存有界）。
        _g16_sink = getattr(self, '_g16_stage_sink', None)
        if _g16_sink is not None:
            _g16_sink.clear()  # 开头清：兜底 profiling off 时的上次残留
            if hasattr(self.remote_selector, '_g16_sink'):
                self.remote_selector._g16_sink = _g16_sink
        _learned = getattr(self.remote_selector, 'g16_treatment', '') in (
            'learned', 'learned_v2', 'csnet')
        if _learned:
            # learned scorer：软权重 w [B,S,M] + 候选索引 cand_idx [B,S,M] → 软加权聚合
            # （梯度回流 scorer，推理可退化为硬 top-k）
            # center_xyz 用 fps_idx 取准确中心坐标（learned_v2 的时间跨度/XY 距离信号）
            center_xyz = index_points(xytp[:, :, 0:3], fps_idx)        # [B, S, 3]
            w, cand_idx = self.remote_selector(centroid_feat, points, idx, anchor,
                                               xytp[:, :, 0:3], center_xyz=center_xyz)
            cand_points = index_points(points, cand_idx)               # [B, S, M, C]
            # 加权平均（除以 Σw 归一化，量级对齐 hyper_agg 的 LayerNorm 输出；
            # 否则 one-hot 求和 = 16×特征量级，会淹没拼接里的 local 部分）
            remote_feat = torch.einsum('bsm,bsmc->bsc', w, cand_points) / \
                w.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, S, C]
        else:
            remote_idx = self.remote_selector(centroid_feat, points, idx, anchor,
                                              xytp[:, :, 0:3])  # [B, S, G]
            # selector 缓存的记录里补填 centroid（fps_idx 只在 grouper 层可见）
            if _g16_sink is not None and len(_g16_sink) > 0:
                rec = _g16_sink[-1]
                if rec.get('centroid') is None and rec.get('stage') == getattr(
                        self.remote_selector, 'stage_idx', None):
                    rec['centroid'] = fps_idx.detach()
                    # xytp 用本 stage 完整 4 通道（含 p）——profile agent 吃 [B,N,4]
                    rec['xytp'] = xytp.detach()
            remote_points = index_points(points, remote_idx)               # [B, S, G, C]
            if self.remote_kda_enc is not None:
                # 远程 KDA 时间编码（EventMamba 启发）：残差加法同局部 KDA 接法，
                # KDA 内部按时间戳自动排序，恢复远程点间的时序因果关系
                remote_xytp = index_points(xytp, remote_idx)               # [B, S, G, 4]
                remote_points = remote_points + self.remote_kda_enc(
                    remote_points, remote_xytp[..., 0:1])                  # [B, S, G, C]
            # ---- stochastic depth：训练时以 p_drop 概率逐样本跳过双曲远程聚合支路 ----
            # 展平为 [M, G, C]（M = B*S，每个 FPS 组独立伯努利），跳过的组直接用
            # 残差 x.mean(dim=1)（组均值）代替 hyper_agg 输出；推理（self.training=False）
            # 时不跳过。keep mask [M,1] 与 [M,C] 广播。
            p_drop = self.remote_stochastic_depth
            if self.training and p_drop > 0.0:
                G, Cc = remote_points.shape[2], remote_points.shape[3]
                M = B * S
                flat = remote_points.reshape(M, G, Cc)                     # [M, G, C]
                agg = self.hyper_agg(flat.unsqueeze(1)).squeeze(1)         # [M, C]
                residual = flat.mean(dim=1)                                # [M, C]
                keep = (torch.rand(M, 1, device=flat.device, dtype=flat.dtype) > p_drop)
                remote_feat = torch.where(keep, agg, residual).reshape(B, S, -1)  # [B, S, C]
            else:
                remote_feat = self.hyper_agg(remote_points)                # [B, S, C]
        remote_feat = self.remote_dropout(remote_feat)
        # ---- 门控融合（SECNet 启发，remote_gate=True 时生效）----
        # gate = sigmoid(MLP([remote_feat, centroid_feat]))：由中心点特征决定
        # 远程特征的逐通道通过率，初始化下 gate≈0.88，起步近似原直连行为
        if self.use_remote_gate:
            gate = torch.sigmoid(self.remote_gate(
                torch.cat([remote_feat, centroid_feat], dim=-1)))  # [B, S, C]
            remote_feat = remote_feat * gate
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
