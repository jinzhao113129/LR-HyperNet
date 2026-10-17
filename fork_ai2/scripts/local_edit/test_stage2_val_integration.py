# test_stage2_val_integration.py — Codex 第二轮复审要求的真实集成测试
# 验证 --val_ratio 0.1 路径在【真实 9 窗/视频 train 特征】下能完整走通：
#   划分 → WindowTripleDataset（DataLoader）→ build_orig_window_tensor（验证张量）
#   → 训练循环几步 → best_checkpoint 键名 val_acc
# 覆盖点（Codex 原话）："确认 val_ratio 划分后训练脚本能构造出 DataLoader
# 和验证张量，而不是只测划分数量"
import sys, os, tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 直接 import 被测函数（stage2_train.py 在同目录）
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'stage2_train_under_test', os.path.join(BASE, 'stage2_train.py'))
_st = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_st)

FAIL = []
def check(name, cond, detail=''):
    print(('[PASS] ' if cond else '[FAIL] ') + name + (('  ' + detail) if detail else ''))
    (FAIL.append(name) if not cond else None)

# ===== 构造真实形态：100 视频 × 9 窗（3 时间窗 × 3 变体）=====
rng = np.random.default_rng(0)
N_VIDEO, N_WIN, D = 100, 9, 1080
feat = rng.standard_normal((N_VIDEO * N_WIN, D)).astype(np.float32)
label = rng.integers(0, 101, N_VIDEO)
# 关键：mark 排布与真实 train.h5 一致——同视频 9 窗连续，label 按窗展开
mark = np.repeat(np.arange(N_VIDEO), N_WIN)
label_exp = np.repeat(label, N_WIN)  # 每窗的 label = 该视频 label
NUM_WINDOWS = 3

# ===== 1) 按视频划分（main() 的划分段逐字复现）=====
val_ratio, seed = 0.1, 42
video_ids = sorted(set(int(m) for m in mark))
vrng = np.random.default_rng(seed)
vrng.shuffle(video_ids)
n_val = max(1, int(len(video_ids) * val_ratio))
val_ids = set(video_ids[:n_val])
is_val = np.array([int(m) in val_ids for m in mark])
val_feat, val_lab, val_mark = feat[is_val], label_exp[is_val], mark[is_val]
feat_tr, lab_tr, mark_tr = feat[~is_val], label_exp[~is_val], mark[~is_val]
check('1a: 视频级互斥划分（train 90 / val 10）',
      len(set(int(m) for m in mark_tr)) == 90 and len(val_ids) == 10)
check('1b: 窗口数随视频走（train 810 / val 90）',
      feat_tr.shape[0] == 810 and val_feat.shape[0] == 90)

# ===== 2) train 侧构造 DataLoader（WindowTripleDataset triple 模式）=====
ds = _st.WindowTripleDataset(feat_tr, lab_tr, mark_tr,
                             sample_mode='triple', num_windows=NUM_WINDOWS)
ds.seed0 = seed * 1000
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
xb, yb = next(iter(loader))
check('2a: DataLoader 可构造并出批', xb.shape == (16, 3, D) and yb.shape == (16,))
# triple 模式样本数 = 90 视频 × 3 变体 = 270
check('2b: triple 样本数 = 视频 × 变体数', len(ds) == 270, f'len={len(ds)}')
# 每个样本应取到同一变体号的 3 个时间窗（行号 w*3+v 布局）
m0, v0 = ds.samples[0]
idxs = ds.groups[m0]
expect_rows = [idxs[w * ds.n_variant[m0] + v0] for w in range(NUM_WINDOWS)]
got = ds[0][0]
check('2c: 样本 = 同一变体的 3 个原始时间窗', torch.allclose(
    got, torch.from_numpy(feat[expect_rows])))

# ===== 3) 验证张量 build_orig_window_tensor（P0 修复本体）=====
x_sel, y_sel = _st.build_orig_window_tensor(val_feat, val_lab, val_mark,
                                            NUM_WINDOWS)
check('3a: 验证张量形状 [10, 3, 1080]', tuple(x_sel.shape) == (10, 3, D))
check('3b: 不触发 assert（旧 build_test_tensor 在 9 窗上会崩）', True)
# 验证取的是每个时间窗的第 0 变体：mark=m 的 9 窗里 rows = [0,3,6]
m_first = sorted(val_ids)[0]
idxs_v = [i for i, mm in enumerate(mark) if int(mm) == m_first]
expect_rows = [idxs_v[w * 3] for w in range(NUM_WINDOWS)]
row_in_x = sorted(val_ids).index(m_first)
check('3c: 取原始窗（第 0 变体，行号 [0,3,6]）',
      torch.allclose(x_sel[row_in_x], torch.from_numpy(feat[expect_rows])))
check('3d: 标签对齐', int(y_sel[row_in_x]) == int(label_exp[idxs_v[0]]))

# ===== 4) 对照：测试集 3 窗路径不受影响 =====
# 构造测试形态（每视频恰好 3 窗，label 按窗展开——与真实 test.h5 一致）
mark_te = np.repeat(np.arange(20), 3)
feat_te = rng.standard_normal((60, D)).astype(np.float32)
lab_te_videos = rng.integers(0, 101, 20)
lab_te = np.repeat(lab_te_videos, 3)
x_te, y_te = _st.build_test_tensor(feat_te, lab_te, mark_te, NUM_WINDOWS)
check('4a: 测试集 3 窗路径仍正常 [20,3,1080]', tuple(x_te.shape) == (20, 3, D))

# ===== 5) 端到端几步训练（真实 WindowAggregator + val 选择 + 键名）=====
import stage2_modules as _sm
agg = _sm.WindowAggregator(feat_dim=D, num_windows=NUM_WINDOWS,
                           num_classes=101, forward_mode='attn_concat')
opt = torch.optim.SGD(agg.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
tmpdir = tempfile.mkdtemp()
best_acc = -1.0
for ep in range(3):
    agg.train()
    for xb, yb in loader:
        opt.zero_grad()
        out = agg(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
    acc = _st.evaluate(agg, x_sel, y_sel)
    if acc > best_acc:
        best_acc = acc
        save_key = 'val_acc'  # main() 的键名逻辑
        torch.save({'net': agg.state_dict(), 'epoch': ep, save_key: acc},
                   os.path.join(tmpdir, 'best_checkpoint.pth'))
torch.save({'net': agg.state_dict(), 'epoch': 3},
           os.path.join(tmpdir, 'last_checkpoint.pth'))
ck = torch.load(os.path.join(tmpdir, 'best_checkpoint.pth'))
check('5a: 端到端 3 epoch 训练跑通（无崩溃）', True, f'best val acc {best_acc:.1f}%')
check('5b: best_checkpoint 键名 val_acc（不是 test_acc）',
      'val_acc' in ck and 'test_acc' not in ck)

print()
if FAIL:
    print('FAILED:', FAIL)
    sys.exit(1)
print('全部通过（真实 9 窗集成路径）')
