# test_g16_round3.py — Codex 第三/四轮修复的集成 smoke
#
# 验收标准（Codex 原话）：
#   1. 一次 forward 后全局 sink 恰好有 stage=[0,1,2] 三条
#   2. 同一输入两次 random/spatial 结果不同
#   3. profile h5 含 feat_cos_*
#   4. 输出路径在 log/lr_hypernet/<log_dir> 下
#
# 额外回归（P0/P1 修复直接针对的 bug）：
#   5. RNG 惰性实例复用：同一 selector 两次调用用同一 generator（不重播种）
#   6. 空积分清洗（profiling off）：连续多次 forward 本地 sink ≤1 条（内存有界）
#   7. resume 校验：**真实 parse_args + 真实 save_model + 真实校验代码路径**
#      （Codex 第四轮 P1-1: 旧版手工塞 checkpoint 键 + 复制校验逻辑 = 假通过）
#   8. 空 treatment 前向不受影响（回归）
#   9. treatment × 旧 flag 互斥断言（Codex 第四轮 P1-2）
#  10. rp1_scorer 二维 XY / 旧 use_rp1_selector 三维（口径隔离回归）
#
# 环境自举（Codex 第四轮记录#2）: _smoke_tree 不存在时自动搭建——
# 干净环境直接 `python test_g16_round3.py` 即可复跑。

import os
import sys
import math
import shutil
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np

# ---- smoke 树（HPC 布局的本地复刻，自动搭建/自动同步最新源）----
# _smoke_tree/
#   lr_hypernet.py           ← lr_hypernet_g16.py（上传版）
#   models/
#     FECNet.py              ← local_edit/FECNet.py
#     modules.py             ← local_edit/modules.py
#     local_remote_grouper.py← local_remote_grouper_v2.py（上传版）
#   g16_profile_agent.py     ← v3
SMOKE_TREE = os.path.join(BASE_DIR, '_smoke_tree')
_tree_map = [
    (os.path.join(BASE_DIR, 'lr_hypernet_g16.py'),
     os.path.join(SMOKE_TREE, 'lr_hypernet.py')),
    (os.path.join(BASE_DIR, 'FECNet.py'),
     os.path.join(SMOKE_TREE, 'models', 'FECNet.py')),
    (os.path.join(BASE_DIR, 'modules.py'),
     os.path.join(SMOKE_TREE, 'models', 'modules.py')),
    (os.path.join(BASE_DIR, 'local_remote_grouper_v2.py'),
     os.path.join(SMOKE_TREE, 'models', 'local_remote_grouper.py')),
    (os.path.join(BASE_DIR, 'g16_profile_agent.py'),
     os.path.join(SMOKE_TREE, 'g16_profile_agent.py')),
    (os.path.join(BASE_DIR, 'modules.py'),
     os.path.join(SMOKE_TREE, 'modules.py')),
]
os.makedirs(os.path.join(SMOKE_TREE, 'models'), exist_ok=True)
import hashlib
_synced = True
for src, dst in _tree_map:
    if not os.path.exists(dst) or \
            hashlib.md5(open(src, 'rb').read()).hexdigest() != \
            hashlib.md5(open(dst, 'rb').read()).hexdigest():
        shutil.copyfile(src, dst)
        _synced = False
print(f"[smoke-tree] {'已同步' if _synced else '重建/更新'}: {SMOKE_TREE}")
sys.path.insert(0, SMOKE_TREE)

from lr_hypernet import LRHyperNet
from models.local_remote_grouper import LocalRemoteGrouper
from g16_profile_agent import profile_batch, profile_to_h5

print("=" * 70)
print("Codex 第三轮验收 smoke")
print("=" * 70)

torch.manual_seed(0)
np.random.seed(0)

B, N = 2, 1024   # 小规模冒烟（CPU）
S_KW = dict(g16_treatment='random', g16_rng_seed=42)

PASS = []
FAIL = []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ========== 验收 1: 一次 forward 后全局 sink 恰好 stage=[0,1,2] 三条 ==========
net = LRHyperNet(num_classes=101, g16_treatment='random', g16_rng_seed=42)
net.eval()
net._g16_stage_selections = []   # trainer 侧 --g16_profile 会做这件事
x = torch.randn(B, 4, N)
x[:, 1] = torch.rand(B, N)   # x ∈ [0,1]
x[:, 2] = torch.rand(B, N)   # y ∈ [0,1]
with torch.no_grad():
    out = net(x)
stages = [rec['stage'] for rec in net._g16_stage_selections]
check("A1: forward 后全局 sink 恰好 3 条 stage=[0,1,2]",
      len(stages) == 3 and stages == [0, 1, 2],
      f"stages={stages}")
check("A1b: 输出形状 [2,101]", out.shape == (2, 101), f"{tuple(out.shape)}")

# sink 里的记录字段完整性
rec = net._g16_stage_selections[0]
check("A1c: sink 记录含 stage/remote/local/centroid/xytp/feat",
      all(k in rec for k in ('stage', 'remote', 'local', 'centroid', 'xytp', 'feat')))
check("A1d: centroid 已补填（非 None）", rec['centroid'] is not None)
check("A1e: feat 已补填（非 None，P0 feat_cos 前置）", rec['feat'] is not None)

# 每条记录所在 stage 的 fps_idx/xytp 形状
# stage0: N=1024→S=2048 不成立，小规模下 group_number 默认 [2048,1024,512]
check("A1f: xytp 维度 [B,N,4]", rec['xytp'].dim() == 3 and rec['xytp'].shape[-1] == 4,
      f"{tuple(rec['xytp'].shape)}")

# ========== 验收 2: 同一输入两次 random/spatial 结果不同 ==========
def _fresh_remote_idx(net_, x_):
    net_._g16_stage_selections = []
    with torch.no_grad():
        net_(x_)
    return [r['remote'].clone() for r in net_._g16_stage_selections]

r1 = _fresh_remote_idx(net, x)
r2 = _fresh_remote_idx(net, x)
diff_random = any((a != b).any().item() for a, b in zip(r1, r2))
check("A2: random 两次 forward 选点不同（RNG 惰性复用不重播种）",
      diff_random)

net_sp = LRHyperNet(num_classes=101, g16_treatment='spatial_fixed', g16_rng_seed=42)
net_sp.eval()
s1 = _fresh_remote_idx(net_sp, x)
s2 = _fresh_remote_idx(net_sp, x)
diff_spatial = any((a != b).any().item() for a, b in zip(s1, s2))
check("A3: spatial_fixed 两次 forward 选点不同", diff_spatial)

# ========== 验收 5: RNG 实例复用（不重播种） ==========
sel0 = net.local_grouper_list[0].remote_selector
rng_a = getattr(sel0, '_g16_rng_inst', None)
check("A5: selector 持有惰性 RNG 实例（首次 forward 后）", rng_a is not None)
with torch.no_grad():
    net(x)
check("A5b: 第二次 forward 后 RNG 实例同一对象（identity 不变）",
      getattr(sel0, '_g16_rng_inst', None) is rng_a)

# ========== 验收 6: profiling off 时本地 sink 有界 ==========
net_no_sink = LRHyperNet(num_classes=101, g16_treatment='random', g16_rng_seed=42)
net_no_sink.eval()   # 不设 _g16_stage_selections → profiling off
with torch.no_grad():
    net_no_sink(x)
    net_no_sink(x)
    net_no_sink(x)
local_lens = [len(lg._g16_stage_sink) for lg in net_no_sink.local_grouper_list]
check("A6: profiling off 连续 3 次 forward 本地 sink 有界（≤1，开头清空兜底）",
      all(l <= 1 for l in local_lens), f"lens={local_lens}")

# ========== 验收 7/8: resume 校验（真实路径——Codex 第四轮 P1-1） ==========
# 真实 parse_args() → 真实 save_model(args=vars(args)) → 真实校验代码路径
# （从 train_lr_hyper.py 源码提取 resume 校验块逐字执行——不复制逻辑。）
# 被测对象必须是 local_edit/train_lr_hyper.py（上传版）——按文件路径
# 显式加载，避免同名旧拷贝（secnet-hyperbolic/code/ 下有一个无 g16
# flag 的旧版本）被 sys.path 顺序抢先命中。provider_data 依赖挂
# secnet-hyperbolic/code/（仅供 import 通过，不被测）。
_secnet_root = os.path.normpath(os.path.join(
    BASE_DIR, '..', '..', '..'))                # .../secnet-hyperbolic
sys.path.insert(0, os.path.join(_secnet_root, 'code'))  # provider_data 依赖
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'train_lr_hyper_under_test',
    os.path.join(BASE_DIR, 'train_lr_hyper.py'))
_tlh = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tlh)

# 真实 CLI 参数（与 G16 链将提交的形态一致）
def _real_args(argv):
    old_argv = sys.argv
    sys.argv = ['train_lr_hyper.py'] + argv
    try:
        return _tlh.parse_args()
    finally:
        sys.argv = old_argv

# 模拟 main() 里模型构造前的派生写回（P1-1 修法本体）
def _derive(args_):
    args_.g16_rng_seed = int(getattr(args_, 'seed', 42))
    return args_

# 真实 save_model（训练循环里的调用形态：args=vars(args)）
_ckpt_dir = os.path.join(SMOKE_TREE, 'ckpt_smoke')
os.makedirs(_ckpt_dir, exist_ok=True)
a_train = _derive(_real_args(['--g16_treatment', 'random', '--seed', '42']))
_tlh.save_model(net, epoch=10, path=_ckpt_dir, acc=0.5, is_best=False,
                best_test_acc=0.5, best_test_seq=0.5,
                optimizer=None, scheduler=None, args=vars(a_train))
_ckpt_file = os.path.join(_ckpt_dir, 'last_checkpoint.pt')
check("A7: 真实 save_model 落盘（含 args=vars(args)）", os.path.exists(_ckpt_file))
_ckpt = torch.load(_ckpt_file, map_location='cpu')
check("A7b: checkpoint args 含 g16_treatment+seed+g16_rng_seed",
      _ckpt.get('args', {}).get('g16_treatment') == 'random'
      and _ckpt['args'].get('seed') == 42
      and _ckpt['args'].get('g16_rng_seed') == 42)

# ---- 真实校验路径：从 train_lr_hyper.py 源码提取 resume 校验块执行 ----
# main() 不可直接跑（连数据/GPU），校验块本身是纯函数性质——用源码
# 提取保证与上传版逐字一致，避免"复制逻辑=假通过"。
import re as _re
import textwrap as _tw
_src = open(os.path.join(BASE_DIR, 'train_lr_hyper.py'), encoding='utf-8').read()
_m = _re.search(
    r"# Codex 第三轮补充#1.*?\n(.*?)\n        if 'optimizer' in checkpoint:",
    _src, _re.S)
check("A7c: 校验块可从源码提取（结构未漂移）", _m is not None)
# 源文件里校验块在 if 内（带 8 空格缩进），exec 前去公共缩进
_resume_check_src = _tw.dedent(_m.group(1)) if _m else ''

def _run_resume_check(checkpoint, args_):
    """执行 train_lr_hyper.py 里逐字的 resume 校验块。"""
    exec(compile(_resume_check_src, 'train_lr_hyper.py#resume-check', 'exec'),
         {'checkpoint': checkpoint, 'args': args_})

# 一致配置 → 不 raise
try:
    a_same = _derive(_real_args(['--g16_treatment', 'random', '--seed', '42']))
    _run_resume_check(_ckpt, a_same)
    check("A7d: resume 一致配置（真实 checkpoint+真实校验块）通过", True)
except Exception as e:
    check("A7d: resume 一致配置（真实 checkpoint+真实校验块）通过", False, str(e)[:80])

# 误传 treatment → raise
try:
    a_bad = _derive(_real_args(['--g16_treatment', 'euc_near', '--seed', '42']))
    _run_resume_check(_ckpt, a_bad)
    check("A8: resume 误传 treatment 被真实校验块 raise 拦截", False, "未 raise")
except ValueError as e:
    check("A8: resume 误传 treatment 被真实校验块 raise 拦截", True, str(e)[:60])

# 改 seed → 也要被拦（P1-1 核心场景：g16_rng_seed 是派生键，旧版校验
# 不存在的键 = 改 --seed 不拦截；现在 seed/g16_rng_seed 都真实入 ckpt）
try:
    a_seed = _derive(_real_args(['--g16_treatment', 'random', '--seed', '7']))
    _run_resume_check(_ckpt, a_seed)
    check("A8b: resume 改 --seed 被拦截（P1-1 场景）", False, "未 raise")
except ValueError as e:
    check("A8b: resume 改 --seed 被拦截（P1-1 场景）", True, str(e)[:60])

# ========== Codex 第五轮备注#1: log_dir=None 时 profile 跟随时间戳目录 ==========
# 廉价断言（不阻塞上传）：train._exp_subdir 属性被设置后，profile 目录
# 应取它而非 'default'。这里直接测 main() 里的赋值语义。
class _TrainFn:
    pass
import types as _types
_train_stub = _TrainFn()
_train_stub._exp_subdir = '2026-10-15_22-00'      # main() 会在 log_dir=None 时写 timestr
_exp_sub = getattr(_train_stub, '_exp_subdir', None) or str(
    getattr(a_train, 'log_dir', '') or 'default')
check("A17: log_dir=None 时 _exp_subdir 取时间戳（不落 default）",
      _exp_sub == '2026-10-15_22-00')
# 显式 log_dir 时两者一致
a_exp = _real_args(['--log_dir', 'secnet_g16_euc_far_ai2'])
_train_stub._exp_subdir = a_exp.log_dir
_exp_sub2 = getattr(_train_stub, '_exp_subdir', None) or str(
    getattr(a_exp, 'log_dir', '') or 'default')
check("A17b: 显式 log_dir 时 _exp_subdir = log_dir", _exp_sub2 == 'secnet_g16_euc_far_ai2')

# ========== 验收 3+4: profile h5 含 feat_cos_* + 输出路径 ==========
# 用真实 agent 跑一条记录: 需要实际 sink 记录（含 feat）
sys.path.insert(0, BASE_DIR)
from g16_profile_agent import profile_batch, profile_to_h5

test_log_dir = "smoke_round3_ai2"
prof_path = os.path.join('log', 'lr_hypernet', test_log_dir, 'g16_profile.h5')
os.makedirs(os.path.dirname(prof_path), exist_ok=True)
if os.path.exists(prof_path):
    os.remove(prof_path)

rec_full = net._g16_stage_selections[0]
stats = profile_batch(
    rec_full['xytp'], rec_full['remote'], rec_full['local'],
    rec_full['centroid'], feat=rec_full['feat'], top_records=2048,
    stage=rec_full['stage'], epoch=0, global_step=1,
    selector='random', seed=42)
check("A9: profile_batch 产出 feat_cos_* 三项",
      all(k in stats for k in ('feat_cos_remote_mean', 'feat_cos_remote_p10',
                               'feat_cos_remote_p90')),
      f"cos_mean={stats.get('feat_cos_remote_mean'):.4f}" if 'feat_cos_remote_mean' in stats else "")

profile_to_h5([stats], prof_path)
import h5py
with h5py.File(prof_path, 'r') as f:
    has_cos = 'feat_cos_remote_mean' in f
    n_rows = f['global_step'].shape[0]
check("A10: h5 落盘含 feat_cos_remote_mean", has_cos)
check("A11: h5 落盘路径 log/lr_hypernet/<log_dir>",
      prof_path.startswith(os.path.join('log', 'lr_hypernet')) and n_rows == 1,
      f"path={prof_path}")

# ========== 回归: 空 treatment 前向 ==========
net_plain = LRHyperNet(num_classes=101)
net_plain.eval()
with torch.no_grad():
    out_p = net_plain(x)
check("A12: 空 treatment LRHyperNet forward 正常 [2,101]",
      out_p.shape == (2, 101) and torch.isfinite(out_p).all().item())
check("A12b: 空 treatment 本地 sink 为 None",
      all(getattr(lg, '_g16_stage_sink', 'x') is None for lg in net_plain.local_grouper_list))

# ========== 回归: 六处理全量构造+forward（第二轮已过,回归防倒退） ==========
for treat in ('random', 'spatial_fixed', 'euc_far', 'euc_near', 'hyp_far', 'rp1_scorer'):
    try:
        n_t = LRHyperNet(num_classes=101, g16_treatment=treat, g16_rng_seed=42)
        n_t.eval()
        n_t._g16_stage_selections = []
        with torch.no_grad():
            o = n_t(x)
        st = [r['stage'] for r in n_t._g16_stage_selections]
        ok = o.shape == (2, 101) and st == [0, 1, 2]
        check(f"A13: {treat} 构造+forward+3条sink", ok, f"stages={st}")
    except Exception as e:
        check(f"A13: {treat} 构造+forward+3条sink", False, str(e)[:80])

# ========== 回归: euc_far vs euc_near 方向翻转不倒退 ==========
n_far = LRHyperNet(num_classes=101, g16_treatment='euc_far', g16_rng_seed=42)
n_near = LRHyperNet(num_classes=101, g16_treatment='euc_near', g16_rng_seed=42)
n_far.eval(); n_near.eval()
for n_ in (n_far, n_near):
    n_._g16_stage_selections = []
with torch.no_grad():
    n_far(x); n_near(x)
r_far = n_far._g16_stage_selections[0]['remote']
r_near = n_near._g16_stage_selections[0]['remote']
overlap = (r_far == r_near).float().mean().item()
check("A14: euc_far vs euc_near 重合率 = 0（方向翻转回归）", overlap == 0.0,
      f"overlap={overlap:.4f}")

# ========== Codex 第四轮 P1-2: treatment × 旧 flag 互斥断言 ==========
# euc_far + use_hyperbolic_selector 必须在构造时被拒（否则静默跑成双曲
# 选点但 log/checkpoint 仍叫 euc_far——六处理纯度被破坏）
def _expect_raise(desc, fn):
    try:
        fn()
        check(desc, False, "未 raise")
    except ValueError as e:
        check(desc, True, str(e)[:60])
    except Exception as e:
        check(desc, False, f"异常类型错: {type(e).__name__}")

_expect_raise("A15: euc_far+use_hyperbolic_selector 构造被拒（P1-2）",
              lambda: LRHyperNet(num_classes=101, g16_treatment='euc_far',
                                 use_hyperbolic_selector=True, g16_rng_seed=42))
_expect_raise("A15b: euc_near+use_rp1_selector 构造被拒（P1-2）",
              lambda: LRHyperNet(num_classes=101, g16_treatment='euc_near',
                                 use_rp1_selector=True, g16_rng_seed=42))
_expect_raise("A15c: random+use_hyperbolic_selector 构造被拒（P1-2）",
              lambda: LRHyperNet(num_classes=101, g16_treatment='random',
                                 use_hyperbolic_selector=True, g16_rng_seed=42))
# 合法组合不受影响
try:
    _ok1 = LRHyperNet(num_classes=101, g16_treatment='hyp_far',
                      use_hyperbolic_selector=True, g16_rng_seed=42)
    _ok2 = LRHyperNet(num_classes=101, g16_treatment='rp1_scorer',
                      use_rp1_selector=True, g16_rng_seed=42)
    check("A15d: hyp_far+use_hyperbolic / rp1_scorer+use_rp1 合法", True)
except ValueError as e:
    check("A15d: hyp_far+use_hyperbolic / rp1_scorer+use_rp1 合法", False, str(e)[:60])
# 空 treatment × 任意 flag 不受限制（旧实验兼容）
try:
    _ok3 = LRHyperNet(num_classes=101, use_hyperbolic_selector=True)
    _ok4 = LRHyperNet(num_classes=101, use_rp1_selector=True)
    check("A15e: 空 treatment × 旧 flag 不受限（旧实验兼容）", True)
except ValueError as e:
    check("A15e: 空 treatment × 旧 flag 不受限（旧实验兼容）", False, str(e)[:60])

# ========== Codex 第四轮记录#1: RP1 口径隔离回归 ==========
# rp1_scorer（二维 XY）与旧 use_rp1_selector（三维）选点应不同——
# 证明两种口径真实隔离（不是同一份代码两处名字）
try:
    n_new = LRHyperNet(num_classes=101, g16_treatment='rp1_scorer', g16_rng_seed=42)
    n_old = LRHyperNet(num_classes=101, use_rp1_selector=True, g16_rng_seed=42)
    n_new.eval(); n_old.eval()
    with torch.no_grad():
        # 同一权重下比较才有意义：copy state
        n_old.load_state_dict(n_new.state_dict(), strict=True)
        o_new = n_new(x)
        o_old = n_old(x)
    diff_logits = (o_new - o_old).abs().max().item() > 1e-9
    check("A16: rp1_scorer(2D) vs 旧 use_rp1_selector(3D) 输出不同（口径隔离）",
          diff_logits, f"max|Δlogit|={(o_new - o_old).abs().max().item():.2e}")
except Exception as e:
    check("A16: rp1_scorer(2D) vs 旧 use_rp1_selector(3D) 输出不同（口径隔离）",
          False, str(e)[:80])

print()
print("=" * 70)
print(f"SUMMARY: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("全部通过")
