# g16_profile_agent.py v3 — G16 训练时选择分布诊断（Codex P1 审查修复版）
#
# v1→v2 修复（Codex 2026-10-15 审查 8 项）+ v3 修复（Codex 复审 3 项：advanced indexing/RNG 隔离/H5 schema 校验）：
# 1. profile_to_h5 覆盖旧文件 → 分段命名（log_dir+seg 前缀）+ 存在即 append
# 2. 缺 stage/epoch/global_step/selector/seed 字段 → 全部补齐入 H5
# 3. generator 消耗首条 → 强制 materialize 成 list
# 4. top_records 未用 → 随机子采样（上限 top_records 个 center）再统计，
#    避免生产规模 [B=32,S=2048,G=16]≈百万级元素的全量 quantile/item 同步
# 5. occupancy 非 gain → 补 local occupancy 与 local∪remote occupancy，输出
#    remote_occupancy / union_occupancy / occupancy_gain 三值（改名+真 gain）
# 6. 确定性中心抽样 bias → torch.randperm 随机抽 top_records 个 center
# 7. 固定 N step 采样与 dataloader 顺序 alias → Bernoulli 采样（调用方传
#    trigger，训练侧用 random()<p 或每 epoch 随机 batch 决定）
# 8. 输入断言 → 索引 dtype/设备/形状全校验
#
# 用法：训练循环内按 Bernoulli(p≈0.01) 触发调用（非固定步数）。
# 2026-10-15 v3（Codex 复审通过，可接入训练）。

import numpy as np
import torch


@torch.no_grad()
def profile_batch(xytp, remote_idx, local_idx, centroid_idx,
                  feat=None, top_records=2048,
                  stage=0, epoch=0, global_step=0, selector='', seed=None):
    """采集本 batch 选中远程点的分布统计（RP1 原文 §4.1 特征代理版）。

    Returns:
        dict[str, float | int] —— 统计 + 元数据（stage/epoch/step/selector/seed）
    """
    # ---- 8. 输入断言 ----
    assert top_records > 0, f"top_records 须 >0，收到 {top_records}（传 0 会使统计全 NaN）"
    assert xytp.dim() == 3 and xytp.shape[-1] == 4, f"xytp [B,N,4] 现为 {tuple(xytp.shape)}"
    B, N, _ = xytp.shape
    assert remote_idx.dim() == 3, f"remote_idx [B,S,G] 现为 {tuple(remote_idx.shape)}"
    assert local_idx.dim() == 3, f"local_idx [B,S,K] 现为 {tuple(local_idx.shape)}"
    assert centroid_idx.dim() == 2, f"centroid_idx [B,S] 现为 {tuple(centroid_idx.shape)}"
    S = centroid_idx.shape[1]
    G = remote_idx.shape[-1]
    K = local_idx.shape[-1]
    assert remote_idx.shape == (B, S, G) and local_idx.shape == (B, S, K)
    for t, n in [(remote_idx, 'remote'), (local_idx, 'local'), (centroid_idx, 'centroid')]:
        assert t.dtype == torch.long, f"{n}_idx 需 long，现为 {t.dtype}"
        assert t.device == xytp.device, f"{n}_idx 设备 {t.device} != xytp {xytp.device}"

    stats = {'stage': stage, 'epoch': epoch, 'global_step': global_step,
             'selector': selector, 'seed': seed if seed is not None else -1}

    # ---- 4/6. 随机子采样 center（Codex 修复：独立 generator，不消耗全局 RNG）----
    n_centers = B * S
    if n_centers > top_records:
        gen = torch.Generator(device=xytp.device)
        gen.manual_seed(int(global_step) + 1)   # 确定性且与训练 RNG 无关
        sel = torch.randperm(n_centers, device=xytp.device, generator=gen)[:top_records]
        b_idx = sel // S
        s_idx = sel % S
    else:
        b_idx, s_idx = torch.meshgrid(torch.arange(B, device=xytp.device),
                                      torch.arange(S, device=xytp.device), indexing='ij')
        b_idx, s_idx = b_idx.flatten(), s_idx.flatten()

    # gather 到子采样中心粒度（Codex 修复：advanced indexing 替代手工 flatten 偏移——
    # v2 的 (idx + b_idx*N) 在 b_idx 乱序时索引错行，实测 got != expected）
    R = b_idx.numel()
    rem_idx_sel = remote_idx[b_idx, s_idx]      # [R, G]
    loc_idx_sel = local_idx[b_idx, s_idx]       # [R, K]
    cent_idx_sel = centroid_idx[b_idx, s_idx]   # [R]
    bcol = b_idx.unsqueeze(1)                   # [R, 1] 广播用
    rem = xytp[bcol, rem_idx_sel]               # [R, G, 4]
    loc = xytp[bcol, loc_idx_sel]               # [R, K, 4]
    cent = xytp[b_idx, cent_idx_sel]            # [R, 4]

    # ---- 1. XY distance ----
    cent_xy = cent[:, 1:3]
    d_rem_xy = ((rem[:, :, 1:3] - cent_xy.unsqueeze(1)) ** 2).sum(-1).sqrt()   # [R,G]
    d_loc_xy = ((loc[:, :, 1:3] - cent_xy.unsqueeze(1)) ** 2).sum(-1).sqrt()   # [R,K]
    stats['xy_dist_remote_mean'] = d_rem_xy.mean().item()
    stats['xy_dist_remote_p90'] = d_rem_xy.flatten().quantile(0.9).item()
    stats['xy_dist_local_mean'] = d_loc_xy.mean().item()
    stats['xy_dist_ratio_rem_loc'] = (d_rem_xy.mean() / (d_loc_xy.mean() + 1e-8)).item()

    # ---- 2. Temporal gap ----
    cent_t = cent[:, 0]
    dt_rem = (rem[:, :, 0] - cent_t.unsqueeze(1)).abs()
    dt_loc = (loc[:, :, 0] - cent_t.unsqueeze(1)).abs()
    stats['t_gap_remote_mean'] = dt_rem.mean().item()
    stats['t_gap_local_mean'] = dt_loc.mean().item()
    stats['t_gap_ratio'] = (dt_rem.mean() / (dt_loc.mean() + 1e-8)).item()

    # ---- 3. Spatial novelty（rem 到最近 local 的距离）----
    nov = torch.cdist(rem[..., 1:3].reshape(-1, G, 2),
                      loc[..., 1:3].reshape(-1, K, 2)).min(-1).values  # [R,G]
    stats['spatial_novelty_mean'] = nov.mean().item()
    stats['spatial_novelty_p10'] = nov.flatten().quantile(0.1).item()

    # ---- 5. Occupancy（真 gain：local vs local∪remote）----
    grid = 4
    cell_of = lambda pts: (pts[..., 1].clamp(0, 0.999) * grid).long() * grid + \
                          (pts[..., 2].clamp(0, 0.999) * grid).long()
    # 每行 unique 格子数（向量化近似：用 sort 后数差分）
    def count_unique_cells(pts):  # pts [R, M, 4] -> [R]
        c = cell_of(pts)                       # [R, M]
        c_sorted, _ = c.sort(dim=-1)
        changed = torch.cat([torch.ones_like(c_sorted[:, :1]),
                             (c_sorted[:, 1:] != c_sorted[:, :-1]).long()], dim=-1)
        return changed.sum(-1)                 # [R]
    occ_rem = count_unique_cells(rem)          # remote 自己的格数
    occ_loc = count_unique_cells(loc)          # local 自己的格数
    occ_union = count_unique_cells(torch.cat([rem, loc], dim=1))  # 并集
    stats['remote_occupancy_grid4x4_mean'] = occ_rem.float().mean().item()
    stats['local_occupancy_grid4x4_mean'] = occ_loc.float().mean().item()
    stats['union_occupancy_grid4x4_mean'] = occ_union.float().mean().item()
    stats['occupancy_gain_mean'] = (occ_union - occ_loc).float().mean().item()  # Codex #5: 真 gain

    # ---- Polarity agreement / clamp 率 ----
    stats['polarity_agreement'] = (torch.sign(rem[..., 3]) ==
                                   torch.sign(cent[:, 3].unsqueeze(1))).float().mean().item()
    stats['xy_clamp_rate'] = ((rem[..., 1] < 0) | (rem[..., 1] > 1) |
                              (rem[..., 2] < 0) | (rem[..., 2] > 1)).float().mean().item()

    # ---- Feature cosine（可选）----
    if feat is not None:
        assert feat.dim() == 3 and feat.shape[:2] == (B, N)
        # Codex 修复：advanced indexing（同 xytp 的 gather 修法）
        cf = feat[b_idx, cent_idx_sel]                    # [R, C]
        rf = feat[bcol, rem_idx_sel]                      # [R, G, C]
        cos = torch.nn.functional.cosine_similarity(rf, cf.unsqueeze(1), dim=-1)  # [R,G]
        stats['feat_cos_remote_mean'] = cos.mean().item()
        stats['feat_cos_remote_p10'] = cos.flatten().quantile(0.1).item()
        stats['feat_cos_remote_p90'] = cos.flatten().quantile(0.9).item()

    return stats


def profile_to_h5(stats_iter, path):
    """统计序列 → h5。v2: materialize（修复 generator 首条消耗）+ 存在即 append。"""
    stats_list = list(stats_iter)  # 3. 先物化（v1 直接迭代 generator 会丢首条）
    if not stats_list:
        return
    import h5py
    keys = [k for k in stats_list[0] if k not in ('selector',)]
    import os
    if os.path.exists(path):      # 1. append 而非覆盖
        with h5py.File(path, 'a') as f:
            n0 = f['global_step'].shape[0] if 'global_step' in f else 0
            # 5b. Codex 修复：schema 一致性校验——本次 keys 必须与现有完全一致
            #（防止第一次传 feat 第二次没传导致数据集长度错位/v1 旧 schema 混入）
            existing = set(f.keys())
            incoming = set(keys) | {'selector'}
            if existing != incoming:
                raise ValueError(
                    f"H5 schema 不一致，拒绝 append：\n  现有 {sorted(existing)}\n"
                    f"  本次 {sorted(incoming)}\n（如首次调用传了 feat、本次没传，"
                    f"或混用旧 schema 文件——统一调用参数或换新文件。）")
            # 现有各 dataset 长度必须一致（防错位追加）
            lens = {k: f[k].shape[0] for k in existing}
            if len(set(lens.values())) != 1:
                raise ValueError(f"H5 现有 dataset 长度不一致，文件已损坏：{lens}")
            for k in keys:
                arr = np.array([s[k] for s in stats_list],
                               dtype=(np.int64 if isinstance(stats_list[0][k], int) else np.float32))
                if k not in f:
                    f.create_dataset(k, data=arr, maxshape=(None,), chunks=True)
                else:
                    d = f[k]
                    d.resize(n0 + arr.shape[0], axis=0)
                    d[n0:] = arr
            # selector 是字符串 → 存为可变长字符串追加
            sels = [str(s.get('selector', '')) for s in stats_list]
            if 'selector' not in f:
                dt = h5py.string_dtype()
                f.create_dataset('selector', data=sels, maxshape=(None,), dtype=dt,
                                 chunks=True)
            else:
                d = f['selector']
                d.resize(n0 + len(sels), axis=0)
                d[n0:] = sels
    else:
        with h5py.File(path, 'w') as f:
            for k in keys:
                arr = np.array([s[k] for s in stats_list],
                               dtype=(np.int64 if isinstance(stats_list[0][k], int) else np.float32))
                f.create_dataset(k, data=arr, maxshape=(None,), chunks=True)
            sels = [str(s.get('selector', '')) for s in stats_list]
            f.create_dataset('selector', data=sels, maxshape=(None,),
                             dtype=h5py.string_dtype(), chunks=True)
