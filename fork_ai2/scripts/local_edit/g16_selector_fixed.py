# g16_selector_fixed.py — G16 远程选点修正版（独立实现，2026-10-14 ai2/Claude）
#
# 背景（Codex 独立审查 2026-10-14 发现，见 fork_ai2/codex_review_20261014.md #8/#9）：
# 1. 列错位：LocalRemoteGrouper.forward 传给 RemoteSelector 的 all_xyz 是
#    xytp[:, :, 0:3] = (t, x, y)，而 _spatial_uniform_select 把 [..., 0]/[..., 1]
#    当作 x/y 分格——实际按 t×x 分格，y 被忽略。
# 2. 局部泄漏：spatial_uniform 分支缺 random 分支那样的候选并集层 local-membership
#    兜底（尾块 c < G+K 时块内 topk 会把 -inf 排除项带入 cand_idx）。
#
# 本文件提供修正版 _spatial_uniform_select_fixed + RemoteSelectorSpatialFixed
# （修正 spatial_uniform 路径的独立 RemoteSelector 副本），供 G16 对比实验使用。
# 不修改现有 local_remote_grouper.py（运行中作业依赖它）。
#
# 坐标口径约定（与训练数据一致）：xytp 布局 (t, x, y, p)，x=通道1，y=通道2。
# 修正后 spatial_uniform 显式用 x=xytp[..., 1]、y=xytp[..., 2] 分格。

import math
import torch


# ============================================================================
# 双曲距离选点（lorentz，复用 local_remote_grouper 的数值保护实现）
# ============================================================================

def _lorentz_exp_origin_safe(v: torch.Tensor, c: float, eps: float = 1e-6,
                             norm_max: float = 10.0) -> torch.Tensor:
    """指数映射（原点锚定）：欧氏切向量 v ∈ R^d → Lorentz 点 ∈ R^{d+1}。

    与 local_remote_grouper.lorentz_exp_origin（v3.1）公式逐项对齐：
    - 溢出防护：scale = clamp(norm_max/norm, max=1)（超限缩到 norm_max，
      未超限 scale=1 不动——注意方向，勿写成 norm/norm_max）
    - sinh 项：sinh(√c·n)·v/(√c·n)（v/(√c·n) = dir/√c，非 √c·dir）
    - 流形投影兜底：x0 ← sqrt(1/c + ||x[1:]||²)
    独立实现（不 import 运行中作业依赖的模块），供 Codex 逐行比对。
    """
    v = v.float()
    c_t = torch.tensor(float(c), device=v.device, dtype=v.dtype)
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)   # [..., 1]
    scale = torch.clamp(norm_max / norm, max=1.0)               # ≤1，只缩不放
    v_c = v * scale
    norm_c = (norm * scale).clamp_min(eps)                      # = min(||v||, norm_max)
    sqrt_c = torch.sqrt(c_t)
    cosh_term = torch.cosh(sqrt_c * norm_c) / sqrt_c            # [..., 1]
    sinh_term = torch.sinh(sqrt_c * norm_c) * v_c / (sqrt_c * norm_c)  # [..., d]
    out = torch.cat([cosh_term, sinh_term], dim=-1)
    x_rest = out[..., 1:]
    x0 = torch.sqrt((1.0 / c_t) + (x_rest * x_rest).sum(-1, keepdim=True))
    return torch.cat([x0, x_rest], dim=-1)


def _lorentz_distance_safe(x: torch.Tensor, y: torch.Tensor, c: float,
                           eps: float = 1e-6) -> torch.Tensor:
    """Lorentz 双曲距离 d(x,y) = (1/√c)·arccosh(-c·⟨x,y⟩_L)。对齐 v3.1。"""
    c_t = torch.tensor(float(c), device=x.device, dtype=x.dtype)
    inner = -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(-1, keepdim=True)
    arg = (-c_t * inner).clamp_min(1.0 + eps)
    return (1.0 / torch.sqrt(c_t)) * torch.acosh(arg)


def hyperbolic_topk_select(cand_feat: torch.Tensor, centroid_feat: torch.Tensor,
                           cand_idx: torch.Tensor, local_idx: torch.Tensor,
                           G: int, c: float = 0.4,
                           near: bool = False) -> torch.Tensor:
    """双曲距离选点（五处理中的 hyp_far / hyp_near）。

    在共享候选池上用 Lorentz 双曲距离打分，取代欧氏 square_distance：
    centroid 与候选都 exp map 到双曲空间后算双曲距离，取 top-G。
    与 RemoteSelector._hyp_dist_chunk（local_remote_grouper.py:219）公式对齐，
    数值保护同 v3.1（norm_max=10 线性缩放 + arccosh clamp）。

    研究假设（g16_selector_design.md §3.2）：特征空间呈树状层级时，双曲距离
    放大"语义层级远但同源"的点——给欧氏最近（KNN 同源、信息增量小）与
    欧氏最远（正交噪声）之间提供几何中间立场。

    Args:
        cand_feat: [B, S, M, C] 候选池点特征（欧氏）
        centroid_feat: [B, S, C] 中心点特征（欧氏）
        cand_idx: [B, S, M] 候选池索引
        local_idx: [B, S, K] 局部邻居索引（池已排除，防御性再排一次）
        G: 远程点数
        c: 曲率（与历史 G16 run 的 curvature_init=0.4 一致）
        near: False=双曲最远（对照我们 topk 语义）；True=双曲最近（对照原文 KNN 语义）
    Returns:
        remote_idx: [B, S, G]（沿 G 维升序）
    """
    B, S, M, C = cand_feat.shape
    # 池已由 build_shared_candidate_pool 保证零局部重叠；防御性兜底
    is_local = (cand_idx.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1)  # [B, S, M]
    # centroid 与候选逐对双曲距离：均 exp map 后 (B*S) 批展平
    centroid_h = _lorentz_exp_origin_safe(centroid_feat.reshape(B * S, 1, C), c)      # [BS, 1, C+1]
    cand_h = _lorentz_exp_origin_safe(cand_feat.reshape(B * S * M, C).unsqueeze(-2), c)  # [BSM, 1, C+1]
    # [BS, M, 1] 广播：cand_h 重塑 [BS, M, C+1]
    cand_h = cand_h.reshape(B * S, M, -1)
    dist = _lorentz_distance_safe(centroid_h, cand_h, c).reshape(B, S, M)
    dist = dist.masked_fill(is_local, float("inf") if near else float("-inf"))
    final = dist.topk(G, dim=-1, largest=not near)[1]  # far=largest, near=smallest
    remote_idx = torch.gather(cand_idx, -1, final)
    return remote_idx.sort(dim=-1)[0]


def euclidean_topk_select(cand_feat: torch.Tensor, centroid_feat: torch.Tensor,
                          cand_idx: torch.Tensor, local_idx: torch.Tensor,
                          G: int, near: bool = False) -> torch.Tensor:
    """欧氏特征距离选点（五处理中的 topk_far / topk_near）。

    topk_near 是原文 Global KNN 语义的池版：选特征距离**最近**的非局部 G 点
    （"特征相关但空间远程"——同一语义对象在别处的证据），与我们现有 topk
    模式（largest=True，最不相似）方向相反。topk_far 复现我们现状。
    全局 KNN 需全部 N 点打分（昂贵，11.38ms 原文数据），此处池版近似。
    """
    # [B, S, M]：((cand - centroid)^2).sum
    dist = ((cand_feat - centroid_feat.unsqueeze(-2)) ** 2).sum(-1)
    is_local = (cand_idx.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1)
    dist = dist.masked_fill(is_local, float("inf") if near else float("-inf"))
    final = dist.topk(G, dim=-1, largest=not near)[1]
    remote_idx = torch.gather(cand_idx, -1, final)
    return remote_idx.sort(dim=-1)[0]


def random_select(cand_idx: torch.Tensor, local_idx: torch.Tensor, G: int) -> torch.Tensor:
    """池版随机选点（五处理中的 random，RP1 随机基线）。

    与原 RemoteSelector random 模式的差异：在共享候选池（M 点）上直接
    随机采样 G 个，而非分块随机收集再并集。共用池隔离池构造差异（Codex #9）。
    """
    B, S, M = cand_idx.shape
    is_local = (cand_idx.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1)
    if ((~is_local).sum(-1) < G).any():
        raise ValueError(f"候选池不足以满足零重叠与 G={G}")
    rand = torch.rand(B, S, M, device=cand_idx.device)
    rand = rand.masked_fill(is_local, float("-inf"))
    final = rand.topk(G, dim=-1)[1]
    return torch.gather(cand_idx, -1, final).sort(dim=-1)[0]


def spatial_uniform_select_fixed(cand_idx: torch.Tensor, xytp: torch.Tensor,
                                 local_idx: torch.Tensor, G: int) -> torch.Tensor:
    """修正版 G16 空间均匀网格选点（Codex 审查缺陷 #8/#9 的修复）。

    与原 _spatial_uniform_select 的两处差异：
    1. 坐标列：显式用 x=xytp[..., 1]、y=xytp[..., 2]（布局 (t,x,y,p)）。
       原版用 [..., 0]/[..., 1]，实际按 (t,x) 分格，y 被忽略。
    2. 局部排除：候选并集层显式剔除 local_idx 成员（与 random 分支一致
       的 is_local 掩码）。原版无此兜底，尾块极端情形下局部邻居可混入。
       修正后的硬保证：选中索引与 local_idx 零重叠（测试断言）。

    Args:
        cand_idx: [B, S, M] 候选并集索引（可含块内 -inf 排除项带入的
            局部邻居索引——本函数负责剔除）
        xytp: [B, N, 4] 全部点坐标 (t, x, y, p)（x,y ∈ [0,1]）
        local_idx: [B, S, K] 各中心点的局部邻居索引（待排除）
        G: 需要选取的远程点数（4×4 网格取 G=16）
    Returns:
        remote_idx: [B, S, G] 选中的远程邻居索引（沿 G 维升序）
    """
    B, S, M = cand_idx.shape
    grid_size = math.ceil(math.sqrt(G))
    n_cells = grid_size * grid_size
    # ---- 修正 1：显式 x/y 列（原版 [..., 0]/[..., 1] 是 t/x）----
    cand_full = index_points_t(xytp, cand_idx)            # [B, S, M, 4]
    eps = 1e-6
    cell_x = (cand_full[..., 1].clamp(0.0, 1.0 - eps) * grid_size).long()
    cell_y = (cand_full[..., 2].clamp(0.0, 1.0 - eps) * grid_size).long()
    cell = cell_x * grid_size + cell_y                    # [B, S, M]
    # ---- 修正 2：候选并集层剔除局部邻居（原版缺失）----
    is_local = (cand_idx.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1)  # [B, S, M]
    # 可行性 guard：剔除后有效候选必须 ≥ G，否则"选 G 个远程且零重叠"
    # 在数学上不可能。原版此情形会静默把 -inf 项选进来（Codex 触到的
    # 局部泄漏），修正版选择响亮失败——调用方保证池够大（生产配置
    # M=256 >> G=16，不会触发）。
    n_valid = (~is_local).sum(-1)                          # [B, S]
    if (n_valid < G).any():
        b, s = (n_valid < G).nonzero()[0].tolist()
        raise ValueError(
            f"候选池不足以同时满足零重叠与 G={G}：(b={b}, s={s}) 有效候选 "
            f"{n_valid[b, s].item()} < G（池 M={M}，局部邻居 K={local_idx.shape[-1]}）。"
            f"请增大候选池或减小 G。")
    rand = torch.rand(B, S, M, device=cand_idx.device)
    rand = rand.masked_fill(is_local, float("-inf"))
    # 局部成员已置 -inf，绝不会被网格赢家或补齐池选中
    priority = rand.clone()
    cell_order = torch.randperm(n_cells, device=cand_idx.device)
    for seq, j in enumerate(cell_order.tolist()):
        in_cell = (cell == j) & (~is_local)               # [B, S, M]
        if not in_cell.any():
            continue
        winner = rand.masked_fill(~in_cell, float("-inf")).argmax(-1)   # [B, S]
        has = in_cell.any(-1)                             # [B, S]
        win_priority = 2.0 - seq / n_cells                # ∈ (1, 2]
        old = priority.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
        priority.scatter_(
            -1, winner.unsqueeze(-1),
            torch.where(has, torch.full_like(old, win_priority), old).unsqueeze(-1))
    final = priority.topk(G, dim=-1)[1]                   # [B, S, G]
    remote_idx = torch.gather(cand_idx, -1, final)
    # 防御断言：零重叠 + 唯一性（测试模式下启用；训练调用方无需重复检查）
    # （断言放测试脚本，这里保持生产路径零开销）
    return remote_idx.sort(dim=-1)[0]


def index_points_t(xytp: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按最后一维索引取坐标。xytp: [B, N, D]，idx: [B, S, M] → [B, S, M, D]。"""
    B, N, D = xytp.shape
    flat = xytp.reshape(B * N, D)
    flat_idx = (idx + torch.arange(B, device=idx.device).unsqueeze(1).unsqueeze(2) * N).reshape(-1)
    return flat[flat_idx].reshape(B, idx.shape[1], idx.shape[2], D)


def build_shared_candidate_pool(xytp: torch.Tensor, local_idx: torch.Tensor,
                                M_per_center: int, G: int, chunk: int = 1024,
                                generator: torch.Generator | None = None) -> torch.Tensor:
    """共享非局部候选池（Codex 交付标准：三选择器共用同一池，隔离池构造差异）。

    每中心点从全 N 个非局部点中均匀采样 M_per_center 个（无放回），
    三种选择器（random / feature-topk / spatial_uniform）都在这个池上操作，
    消除"池构造与选择策略混淆"（Codex #9）。

    内存有界实现（Codex 2026-10-15 复核要求，v3）：
    历史配置 B=8、S=2048、K=24、N=10240 下，旧版一次性
    is_local = [B,S,N] 布尔 + rand = [B,S,N] float32 张量约
    8·2048·10240·(1+4)B ≈ 840 MiB；其中 is_local 由
    (arange(N) == local_idx.unsqueeze(-2)).any(-2) 产生 [B,S,K,N]
    ≈ 3.75 GiB 中间布尔张量——16GB V100 上不可接受。改为按 N 分块：
    每块只分配 [B,S,chunk] 掩码与随机分数，块内 topk-M 索引平移后拼
    并集，最后并集 topk-M。数学等价性：全 N 的均匀无放回 M 样本
    = 先均匀分块收集再在并集上均匀取 M（每点入选概率不变，均为
    M/N；分块收集数 = min(M, N_total_chunks·M) ≥ M，保证池大小）。

    可行性 guard（Codex 要求显式报错）：非局部点总数 N-K 必须 ≥ M，
    否则"每中心 M 个非局部候选"数学不可行——响亮报错而非静默收缩。

    Args:
        xytp: [B, N, 4]（只用形状与设备，不取值）
        local_idx: [B, S, K] 局部邻居索引（池中排除）
        M_per_center: 每中心点的候选池大小 M
        G: 远程点数（校验用，M ≥ G 由调用方保证，此处再校验一次）
        chunk: N 分块大小（默认 1024，与 LocalGrouper KNN 分块一致）
    Returns:
        pool_idx: [B, S, M]
    """
    B, N, _ = xytp.shape
    S, K = local_idx.shape[1], local_idx.shape[2]
    if M_per_center < G:
        raise ValueError(f"候选池 M={M_per_center} 必须 ≥ G={G}")
    n_nonlocal = N - K
    if n_nonlocal < M_per_center:
        raise ValueError(
            f"非局部点总数 N-K = {N}-{K} = {n_nonlocal} < M={M_per_center}："
            f"'每中心 M 个非局部候选'数学不可行。请减小 M 或 K。")
    device = xytp.device
    # ---- 内存有界：按 N 分块，每块掩码 [B,S,c]（bool 8·2048·1024 ≈ 16 MiB）----
    # 注意不能用普通 scatter_ 写 bool：~in_chunk 项写 False 到索引 0 与
    # 真值写 True 竞争（scatter_ 重复索引未定义）。用 scatter_add 整型
    # 累加器（重复索引=累加，无竞争），K=24 次小 op。
    topk_list = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        c = end - start
        local_cnt = torch.zeros(B, S, c, dtype=torch.int32, device=device)
        for k in range(K):
            idx_k = local_idx[:, :, k] - start                    # [B, S]
            valid = (idx_k >= 0) & (idx_k < c)
            local_cnt.scatter_add_(
                -1, idx_k.clamp(0, c - 1).unsqueeze(-1),
                valid.unsqueeze(-1).to(torch.int32))
        local_mask = local_cnt > 0                                # [B, S, c]
        # 每块收集 min(M, c) 个非局部候选
        k_chunk = min(M_per_center, c)
        rand = torch.rand(B, S, c, device=device, generator=generator)
        rand = rand.masked_fill(local_mask, float("-inf"))
        topk_list.append(rand.topk(k_chunk, dim=-1)[1] + start)
    cand = torch.cat(topk_list, dim=-1)   # [B, S, ~M*ceil(N/chunk)]
    # 并集层防御性再排除（有界：cand 仅 ~M·n_chunks 个元素）+
    # 可行性 guard：排除后必须 ≥ M
    rand2 = torch.rand(B, S, cand.shape[-1], device=device, generator=generator)
    is_local = (cand.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1)  # [B,S,Mc]
    rand2 = rand2.masked_fill(is_local, float("-inf"))
    n_valid = (~is_local).sum(-1)                                # [B, S]
    if (n_valid < M_per_center).any():
        b, s = (n_valid < M_per_center).nonzero()[0].tolist()
        raise ValueError(
            f"并集非局部候选不足：(b={b}, s={s}) 有效 {n_valid[b, s].item()} < "
            f"M={M_per_center}（K={K} 局部邻居占满了块内候选）。")
    pool_idx = rand2.topk(M_per_center, dim=-1)[1]
    return torch.gather(cand, -1, pool_idx)
