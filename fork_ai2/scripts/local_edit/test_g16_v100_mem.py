# test_g16_v100_mem.py — Codex 2026-10-15 交接清单第二项：V100 实测峰值显存与延迟
#
# 配置 = 历史训练真实形态：B=8, S=2048, N=10240, K=24, M=256, G=16
# backbone 特征维度 = LRHyperNet stage0（embed Linear(4→64) 后 feature_list[0]=64）
# 中心 = 真实 FPS 形态（用随机中心点索引近似——显存/延迟测量的目的下足够）
#
# 验收（Codex 原话）："用实际 B=8、S=2048、N=10240 在 V100 上测完整前向的
# 峰值显存与延迟，并验证五处理使用相同候选池"
#
# 运行：HPC 计算节点（srun/sbatch），bhli 环境。#SBATCH 部署时用 sbatch。

import sys
import time
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16')

from g16_selector_fixed import (
    random_select, spatial_uniform_select_fixed,
    euclidean_topk_select, hyperbolic_topk_select,
    build_shared_candidate_pool, index_points_t)

FAIL = []

def check(name, cond):
    print(('PASS' if cond else 'FAIL'), '-', name)
    if not cond:
        FAIL.append(name)

assert torch.cuda.is_available(), "需要计算节点 GPU"
dev = torch.device('cuda')
torch.backends.cudnn.benchmark = False

# ---- 真实训练配置 ----
B, S, N, K, M, G, C = 8, 2048, 10240, 24, 256, 16, 64
torch.manual_seed(42)
xytp = torch.rand(B, N, 4, device=dev)
feat = torch.randn(B, N, C, device=dev)   # backbone embed 后 stage0 特征
# 真实 FPS：S=2048 个中心（用随机无重复采样近似 FPS 起始点）
local_idx = torch.stack([torch.randperm(N)[:K] for _ in range(B * S)]).reshape(B, S, K).to(dev)
centroid = feat[:, :S, :]                 # [B, S, 64]（示意：前 S 点作中心）

print(f"[cfg] B={B} S={S} N={N} K={K} M={M} G={G} C={C}")
print(f"[gpu] {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

def peak_reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated() / 2**20

def report(tag, pre_mb):
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[mem] {tag}: peak={peak:.0f} MiB (base {pre_mb:.0f}, Δ={peak-pre_mb:.0f})")

# ---- 1. 池构造（内存有界版） ----
pre = peak_reset()
t0 = time.perf_counter()
pool = build_shared_candidate_pool(xytp, local_idx, M, G)
torch.cuda.synchronize()
dt_pool = time.perf_counter() - t0
report(f"pool v3 (bounded)", pre)
print(f"[lat] pool: {dt_pool*1000:.1f} ms/batch ({dt_pool*1000/B:.2f} ms/sample)")
check('m1: 池形状 [B,S,M]', tuple(pool.shape) == (B, S, M))
ov = (pool.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1).sum().item()
check('m2: 池与 local_idx 零重叠（真实规模）', ov == 0)
dup = (pool.unsqueeze(-1) == pool.unsqueeze(-2)).float().sum(-1).max().item()
check('m3: 池索引唯一', dup == 1.0)

cand_feat = index_points_t(feat, pool)    # [B, S, M, C]
pre = peak_reset()  # cand_feat 已分配，记入 base

# ---- 2. 五处理峰值显存 + 延迟（各 3 次取中位，warmup 1 次） ----
results = {}
for name, fn in [
    ('random', lambda: random_select(pool, local_idx, G)),
    ('spatial_uniform_fixed', lambda: spatial_uniform_select_fixed(pool, xytp, local_idx, G)),
    ('euc_far', lambda: euclidean_topk_select(cand_feat, centroid, pool, local_idx, G, near=False)),
    ('euc_near', lambda: euclidean_topk_select(cand_feat, centroid, pool, local_idx, G, near=True)),
    ('hyp_far', lambda: hyperbolic_topk_select(cand_feat, centroid, pool, local_idx, G, c=0.4)),
]:
    fn()  # warmup
    torch.cuda.synchronize()
    pre = peak_reset()
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    report(f"selector:{name}", pre)
    dt = sorted(times)[1]  # median
    results[name] = r
    print(f"[lat] {name}: {dt*1000:.1f} ms/batch ({dt*1000/B:.2f} ms/sample), median of 3")
    check(f'm4-{name}: 输出形状 [B,S,G]', tuple(r.shape) == (B, S, G))
    o = (r.unsqueeze(-1) == local_idx.unsqueeze(-2)).any(-1).sum().item()
    check(f'm5-{name}: 零局部重叠（真实规模）', o == 0)

# ---- 3. 五处理共用同一池 ----
for name, r in results.items():
    in_pool = (r.unsqueeze(-1) == pool.unsqueeze(-2)).any(-1).all().item()
    check(f'm6-{name}: 选点全部来自共享池', in_pool)

print()
if FAIL:
    print('FAILED:', FAIL)
    sys.exit(1)
print('ALL_TESTS_PASS')
