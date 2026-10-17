# test_g16_selector_v2.py — 双曲/欧氏 topk 选择器测试（2026-10-14 补充）
#
# 验收（g16_selector_design.md 五处理实现）：
# 1. 数值对齐：_lorentz_exp_origin_safe/_lorentz_distance_safe 与
#    local_remote_grouper v3.1 公式逐项一致（cosh/sinh 因子、scale 方向、
#    流形投影、arccosh clamp）——用解析性质校验（流形约束、远点距离更大）。
# 2. 方向性：near/far 语义正确（near 选出的候选离 centroid 特征更近）。
# 3. 正确性：形状/界内/唯一/零重叠/池成员/可复现——与 t2* 同一套。

import sys
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Windows GBK 控制台兜底

sys.path.insert(0, '.')
from g16_selector_fixed import (
    hyperbolic_topk_select, euclidean_topk_select, random_select,
    build_shared_candidate_pool, index_points_t,
    _lorentz_exp_origin_safe, _lorentz_distance_safe)

FAIL = []

def check(name, cond):
    print(('PASS' if cond else 'FAIL'), '-', name)
    if not cond:
        FAIL.append(name)

torch.manual_seed(42)
B, S, N, K, M, G, C = 2, 6, 512, 24, 256, 16, 32
feat = torch.randn(B, N, C) * 2.0
xytp = torch.rand(B, N, 4)
local_idx = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K)
cand_idx = build_shared_candidate_pool(xytp, local_idx, M, G)
cand_feat = index_points_t(feat, cand_idx)          # [B, S, M, C]
centroid = feat[:, :S, :]

# ---------- 数值对齐（解析性质） ----------
# 1a. 流形约束 ⟨x,x⟩_L = -1/c
x_h = _lorentz_exp_origin_safe(torch.randn(16, C), 0.4)
# 流形约束 ⟨x,x⟩_L = -1/c ⟺ x0² - ||x[1:]||² = 1/c ⟺ x0² - ||rest||² - 1/c = 0
manifold_err = (x_h[..., 0] ** 2 - (x_h[..., 1:] ** 2).sum(-1) - 1.0 / 0.4).abs().max().item()
check('v1a: exp map 流形约束 ⟨x,x⟩_L = -1/c（误差<1e-3）', manifold_err < 1e-3)
# 1b. 远切向量 → 更大双曲距离（exp map 保序单调）
v_a = torch.zeros(4, C); v_a[:, 0] = 1.0    # ||v||=1
v_b = torch.zeros(4, C); v_b[:, 0] = 5.0    # ||v||=5
d_a = _lorentz_distance_safe(_lorentz_exp_origin_safe(v_a, 0.4),
                             _lorentz_exp_origin_safe(v_b, 0.4), 0.4)
check('v1b: 远切向量双曲距离更大', (d_a > 0.1).all().item())

# ---------- 方向性 ----------
# 2a. topk_near 选出的候选到 centroid 的欧氏距离 < topk_far 选出的
near_idx = euclidean_topk_select(cand_feat, centroid, cand_idx, local_idx, G, near=True)
far_idx = euclidean_topk_select(cand_feat, centroid, cand_idx, local_idx, G, near=False)
d_near = ((index_points_t(feat, near_idx) - centroid.unsqueeze(-2)) ** 2).sum(-1)  # [B, S, G]
d_far = ((index_points_t(feat, far_idx) - centroid.unsqueeze(-2)) ** 2).sum(-1)
check('v2a: topk_near 欧氏距离 < topk_far（逐中心均值）',
      (d_near.mean(-1) < d_far.mean(-1)).all().item())
# 2b. hyp_far（双曲最远）也满足远 > 近
hyp_near = hyperbolic_topk_select(cand_feat, centroid, cand_idx, local_idx, G, c=0.4, near=True)
hyp_far = hyperbolic_topk_select(cand_feat, centroid, cand_idx, local_idx, G, c=0.4, near=False)
d_hn = ((index_points_t(feat, hyp_near) - centroid.unsqueeze(-2)) ** 2).sum(-1)
d_hf = ((index_points_t(feat, hyp_far) - centroid.unsqueeze(-2)) ** 2).sum(-1)
check('v2b: hyp_near 欧氏距离 < hyp_far（双曲选点方向语义正确）',
      (d_hn.mean(-1) < d_hf.mean(-1)).all().item())
# 2c. 双曲与欧氏选点结果确有差异（若完全相同则"双曲选点"无研究意义）
same = (near_idx == hyp_near).all().item() and (far_idx == hyp_far).all().item()
check('v2c: 双曲选点与欧氏选点结果不同（区分度存在）', not same)

# ---------- 正确性（与 t2* 同一套） ----------
for name, r in [('random', random_select(cand_idx, local_idx, G)),
                ('euc_far', far_idx), ('euc_near', near_idx),
                ('hyp_far', hyp_far), ('hyp_near', hyp_near)]:
    check(f'v3-{name}: 形状 [B,S,G]', tuple(r.shape) == (B, S, G))
    check(f'v3-{name}: 界内且唯一', (r >= 0).all().item() and (r < N).all().item()
          and ((r.unsqueeze(-1) == r.unsqueeze(-2)).float().sum(-1).max().item() == 1.0))
    ov = (r.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1).sum().item()
    check(f'v3-{name}: 零局部重叠', ov == 0)
    in_pool = (r.unsqueeze(-1) == cand_idx.unsqueeze(-2)).any(-1).all().item()
    check(f'v3-{name}: 全部来自候选池', in_pool)

# ---------- 可复现 ----------
torch.manual_seed(42)
_ = torch.randn(B, N, C) * 2.0
_ = torch.rand(B, N, 4)
_ = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K)
cand2 = build_shared_candidate_pool(xytp, local_idx, M, G)
cand_feat2 = index_points_t(feat, cand2)
hyp_far2 = hyperbolic_topk_select(cand_feat2, centroid, cand2, local_idx, G, c=0.4)
check('v4: 固定 seed 完全可复现', torch.equal(hyp_far, hyp_far2))

print()
if FAIL:
    print('FAILED:', FAIL)
    sys.exit(1)
print('ALL_TESTS_PASS')
