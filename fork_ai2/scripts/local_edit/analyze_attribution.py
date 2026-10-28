"""从 point_attribution 的 npz 聚合好点的空间分布与贡献衰减。"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

D = '/home/youxinjiang/zhaojin/logs_run/point_attribution'
files = sorted(glob.glob(os.path.join(D, 'sample*.npz')))

# 去重：同一 sample/center 可能有测试残留，按文件名去重
seen = set()
uniq = []
for f in files:
    key = os.path.basename(f).replace('.npz', '')
    if key not in seen:
        seen.add(key)
        uniq.append(f)

rank_gains = defaultdict(list)
rank_logit = defaultdict(list)
centers = []
for f in uniq:
    d = np.load(f)
    gains = d['gains'].astype(np.float64)
    logit_g = d['logit_gains'].astype(np.float64)
    xyz = d['xyz'].astype(np.float64)  # [16,3] t,x,y
    for i in range(len(gains)):
        rank_gains[i].append(gains[i])
        rank_logit[i].append(logit_g[i])
    xy = xyz[:, 1:]
    # XY 跨度（选中点包围盒对角线）
    span = float(np.sqrt(((xy.max(0) - xy.min(0)) ** 2).sum()))
    # 点对平均距离（手动，不依赖 scipy）
    diff = xy[:, None, :] - xy[None, :, :]
    pd = np.sqrt((diff ** 2).sum(-1))
    n = len(xy)
    iu = np.triu_indices(n, k=1)
    pair_mean = float(pd[iu].mean())
    # 时间跨度
    tspan = float(xyz[:, 0].max() - xyz[:, 0].min())
    centers.append({
        'total_gain': float(gains.sum()),
        'xy_span': span,
        'pair_mean_dist': pair_mean,
        'time_span': tspan,
        'empty_loss': float(d['empty_loss']),
    })

n_c = len(uniq)
print('=== 有效中心数:', n_c, '===')
print()
print('=== 按 rank 的平均边际贡献（16 点贡献衰减）===')
cum = 0.0
for i in range(16):
    v = float(np.mean(rank_gains[i]))
    cum += v
    print(f'  rank {i+1:2d}: CE下降 {v:.4f}  累计 {cum:.4f}  目标logit {np.mean(rank_logit[i]):+.4f}')

print()
tg = np.array([c['total_gain'] for c in centers])
print('=== 每个中心总贡献（16 点合计）===')
print(f'  均值 {tg.mean():.4f}  中位 {np.median(tg):.4f}  q10 {np.quantile(tg,.1):.4f}  q90 {np.quantile(tg,.9):.4f}')
print(f'  正贡献(>0.01)中心占比 {(tg>0.01).mean()*100:.1f}%')

span = np.array([c['xy_span'] for c in centers])
pm = np.array([c['pair_mean_dist'] for c in centers])
ts = np.array([c['time_span'] for c in centers])
print()
print('=== 选中 16 点的空间散布 ===')
print(f'  XY 包围盒跨度: 均值 {span.mean():.4f}  中位 {np.median(span):.4f}')
print(f'  点对平均距离:  均值 {pm.mean():.4f}  中位 {np.median(pm):.4f}')
print(f'  时间跨度:      均值 {ts.mean():.4f}')

# 高贡献中心 vs 低贡献中心，空间散布是否更大
hi = tg > np.median(tg)
print()
print('=== 高贡献中心 vs 低贡献中心的空间散布对比 ===')
print(f'  高贡献中心: 点对均距 {pm[hi].mean():.4f}  时间跨度 {ts[hi].mean():.4f}')
print(f'  低贡献中心: 点对均距 {pm[~hi].mean():.4f}  时间跨度 {ts[~hi].mean():.4f}')

# 与 empty_loss 的关系（模型越没把握的样本，远程点贡献越大？）
el = np.array([c['empty_loss'] for c in centers])
print()
print(f'=== 总贡献与初始损失的相关性: {np.corrcoef(tg, el)[0,1]:.3f} ===')

report = {
    'num_centers': n_c,
    'rank_mean_gain': {str(i+1): float(np.mean(rank_gains[i])) for i in range(16)},
    'total_gain': {'mean': float(tg.mean()), 'median': float(np.median(tg)),
                   'q10': float(np.quantile(tg,.1)), 'q90': float(np.quantile(tg,.9)),
                   'positive_pct': float((tg>0.01).mean()*100)},
    'xy_span_mean': float(span.mean()),
    'pair_dist_mean': float(pm.mean()),
    'time_span_mean': float(ts.mean()),
    'corr_gain_with_empty_loss': float(np.corrcoef(tg, el)[0,1]),
    'high_gain_pair_dist': float(pm[hi].mean()),
    'low_gain_pair_dist': float(pm[~hi].mean()),
}
with open(os.path.join(D, 'attribution_analysis.json'), 'w', encoding='utf-8') as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)
print()
print('saved attribution_analysis.json')
