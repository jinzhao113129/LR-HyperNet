# test_g16_selector_fixed.py — Codex 交付标准的索引级测试（tiny + 生产规模）
#
# 验收标准（codex_review_20261014.md "Correctness gate"）：
# 1. 判别用例：t=x=0.5 固定、y 分 4 格 → x/y 公式给 4 个不同格子；
#    原公式（t/x 分格）给同一格。反向（x 变 y 固定）隔离 x。
# 2. 形状 [B, S, G=16]；索引唯一（无重复选点）；索引界内 [0, N)。
# 3. 与 local_idx 零重叠（局部邻居绝不混入远程点）。
# 4. 固定 seed 完全可复现。
# 5. 生产规模 N=10240：同上全断言 + 报告 occupied-bin 覆盖统计。

import sys
import torch

sys.path.insert(0, '.')
from g16_selector_fixed import spatial_uniform_select_fixed, build_shared_candidate_pool, index_points_t

FAIL = []

def check(name, cond):
    print(('PASS' if cond else 'FAIL'), '-', name)
    if not cond:
        FAIL.append(name)


# ========== 测试 1：Codex 判别用例（tiny）==========
# 4 个点：t=x=0.5，y=0.1/0.3/0.6/0.9 → y 分 4 格
torch.manual_seed(0)
N, G = 8, 16
xytp = torch.zeros(1, N, 4)
xytp[..., 0] = 0.5  # t
xytp[..., 1] = 0.5  # x
ys = [0.1, 0.3, 0.6, 0.9]
for i, y in enumerate(ys):
    xytp[0, i, 2] = y
# 其余 4 个点远离（y=0.05 各处）作填充
xytp[0, 4:, 0] = 0.9; xytp[0, 4:, 1] = 0.9; xytp[0, 4:, 2] = 0.05

grid_size = 4
# x/y 公式（修正版行为）
cell_xy = (xytp[0, :4, 1].clamp(0, 1-1e-6) * grid_size).long() * grid_size + \
          (xytp[0, :4, 2].clamp(0, 1-1e-6) * grid_size).long()
# 原版公式（[0]/[1] = t/x —— 缺陷行为）
cell_tx = (xytp[0, :4, 0].clamp(0, 1-1e-6) * grid_size).long() * grid_size + \
          (xytp[0, :4, 1].clamp(0, 1-1e-6) * grid_size).long()
check('t1a: x/y 公式给 4 个不同 y 格', len(set(cell_xy.tolist())) == 4)
check('t1b: 原版 t/x 公式塌缩到同格（复现 Codex 缺陷）', len(set(cell_tx.tolist())) == 1)

# 反向判别：x 分 4 格、t=y=0.5 → 修正版按 x 分格
xytp2 = torch.zeros(1, N, 4)
xytp2[..., 0] = 0.5; xytp2[..., 2] = 0.5
for i, x in enumerate([0.1, 0.3, 0.6, 0.9]):
    xytp2[0, i, 1] = x
xytp2[0, 4:, 0] = 0.9; xytp2[0, 4:, 1] = 0.05; xytp2[0, 4:, 2] = 0.9
cell_xy2 = (xytp2[0, :4, 1].clamp(0, 1-1e-6) * grid_size).long() * grid_size + \
           (xytp2[0, :4, 2].clamp(0, 1-1e-6) * grid_size).long()
check('t1c: 反向用例 x 分 4 格（隔离 x 轴）', len(set(cell_xy2.tolist())) == 4)


# ========== 测试 2：tiny 级完整 forward（含局部排除）==========
# 池必须足够大：N=64、K=8、M=40，注入污染后有效候选 ≈ 40-5 ≈ 35 > G=16
# （Codex 触到的“9/16 重叠”缺陷 = 池不足时原版静默泄漏；修正版在池不足时
# 响亮报错，池足时零重叠——两种行为都要测）
torch.manual_seed(42)
B, S, N, K, G = 2, 6, 64, 8, 16
xytp = torch.rand(B, N, 4)
xytp[..., 1] = xytp[..., 1].clamp(0, 1)  # x
xytp[..., 2] = xytp[..., 2].clamp(0, 1)  # y
local_idx = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K)
# 候选并集：从全 N 采样（含自然撞上的局部邻居），并显式注入前 3 个为局部
cand = torch.stack([torch.randperm(N)[:40] for _ in range(B * S)]).reshape(B, S, 40)
for b in range(B):
    for s in range(S):
        cand[b, s, 0:3] = local_idx[b, s, 0:3]
r = spatial_uniform_select_fixed(cand, xytp, local_idx, G)
check('t2a: 输出形状 [B,S,G]', tuple(r.shape) == (B, S, G))
check('t2b: 索引界内', (r >= 0).all().item() and (r < N).all().item())
dup = (r.unsqueeze(-1) == r.unsqueeze(-2)).float().sum(-1).max().item()
check('t2c: 索引唯一（沿 G 无重复）', dup == 1.0)
overlap = (r.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1).sum().item()
check('t2d: 与 local_idx 零重叠（污染注入下）', overlap == 0)
in_cand = (r.unsqueeze(-1) == cand.unsqueeze(-2)).any(-1).all().item()
check('t2d2: 选中点全部来自候选池', in_cand)

# 可复现性：完整重放 RNG 序列（local_idx → cand → select 内部三步）
torch.manual_seed(42)
_ = torch.rand(B, N, 4)
_ = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K)
_ = torch.stack([torch.randperm(N)[:40] for _ in range(B * S)]).reshape(B, S, 40)
r2 = spatial_uniform_select_fixed(cand, xytp, local_idx, G)
check('t2e: 固定 seed 完全可复现', torch.equal(r, r2))

# 池不足时必须响亮报错（不静默泄漏）：N=24、K=10、M=24=N 全池
# → 有效 = 24-10 = 14 < G=16，零重叠不可行，guard 必须触发
torch.manual_seed(1)
small_local = torch.stack([torch.randperm(24)[:10] for _ in range(4)]).reshape(1, 4, 10)
small_cand = torch.arange(24).view(1, 1, 24).repeat(1, 4, 1)  # 全 N 做池
small_xytp = torch.rand(1, 24, 4)
guard_fired = False
try:
    spatial_uniform_select_fixed(small_cand, small_xytp, small_local, G)
except ValueError:
    guard_fired = True
check('t2f: 池不足时 guard 响亮报错（复现 Codex 缺陷场景）', guard_fired)


# ========== 测试 3：生产规模 N=10240 ==========
torch.manual_seed(7)
B, S, N, K, G, M = 2, 64, 10240, 24, 16, 256
xytp = torch.rand(B, N, 4)
local_idx = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K)
pool = build_shared_candidate_pool(xytp, local_idx, M, G)
check('t3a: 池形状 [B,S,M]', tuple(pool.shape) == (B, S, M))
pool_overlap = (pool.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1).sum().item()
check('t3b: 池与 local_idx 零重叠', pool_overlap == 0)
r3 = spatial_uniform_select_fixed(pool, xytp, local_idx, G)
check('t3c: 生产规模输出形状', tuple(r3.shape) == (B, S, G))
ov3 = (r3.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1).sum().item()
check('t3d: 生产规模与 local_idx 零重叠', ov3 == 0)
in_pool = (r3.unsqueeze(-1) == pool.unsqueeze(-2)).any(-1).all().item()
check('t3e: 选中点全部来自候选池', in_pool)
dup3 = (r3.unsqueeze(-1) == r3.unsqueeze(-2)).float().sum(-1).max().item()
check('t3f: 生产规模索引唯一', dup3 == 1.0)

# 覆盖统计：4×4 网格占用格数（Codex 要求报告 occupied-bin）
cells = (xytp[:, :, 1].clamp(0, 1-1e-6) * 4).long() * 4 + (xytp[:, :, 2].clamp(0, 1-1e-6) * 4).long()
sel_cells = index_points_t(cells.unsqueeze(-1), r3).squeeze(-1)
occ = (sel_cells.unsqueeze(-1) == torch.arange(16).view(1, 1, 1, 16)).any(-2).float().sum(-1)
print(f'INFO: occupied bins per (b,s): mean={occ.float().mean():.1f}/16, '
      f'min={occ.min():.0f}, max={occ.max():.0f}')

# 复现性（生产级）：完整重放 RNG（local_idx → pool → select）
torch.manual_seed(7)
_ = torch.rand(B, N, 4)
_ = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K)
pool2 = build_shared_candidate_pool(xytp, local_idx, M, G)
r3b = spatial_uniform_select_fixed(pool2, xytp, local_idx, G)
check('t3g: 生产规模固定 seed 可复现', torch.equal(r3, r3b))

# ========== 结果 ==========
print()
if FAIL:
    print('FAILED:', FAIL)
    sys.exit(1)
print('ALL_TESTS_PASS')
