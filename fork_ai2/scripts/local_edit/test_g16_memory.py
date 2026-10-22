"""G16 真实训练显存测试（3090 上跑，Codex P0-2 复审版 2026-10-22）。

相比上一版（B8/accum2 单窗口 + 只打印不判定）的修复：
1. accum 2 -> 4（对齐快测名义 batch32）；
2. 启用 EMA（与 train_frequency_ncaltech.ModelEMA 等价：shadow = float 深拷贝，
   每个 optimizer.step 后 update）；
3. 至少两个 optimizer 更新窗口（覆盖 Adam 状态已分配后的稳态峰值）；
4. 超阈值 exit 1、通过 exit 0（一键脚本可用 $? 判成败）；
5. 打印实际导入模块路径、allocated/reserved 峰值、external0、总显存、判定表达式。

判定口径（Codex 修正）：est_total_used = peak_reserved + external0，
  判定 est_total_used + 安全余量 < total。
  external0 = (total - free0) - reserved0 = 非本进程 allocator 的固定占用
  （其他进程 + CUDA context），不是把 baseline 直接加到 reserved 上重复计数。

每个 treatment 独立子进程（避免上一轮 CUDA 计算图残留污染下一轮峰值）。

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_g16_memory.py
"""
import sys
import subprocess

B, N, ACCUM = 8, 10240, 4
N_WINDOWS = 3            # 至少 2 个 optimizer 更新窗口（Codex 要求）
SAFETY_MARGIN_GB = 1.0   # 安全余量（判定表达式中显式打印）
TREATMENTS = ['hyp_far', 'rp1_scorer']

CHILD = r'''
import sys, torch
sys.path.insert(0, '.')
from FECNet import FECNet

B, N, ACCUM = 8, 10240, 4
N_WINDOWS = 3
SAFETY_MARGIN_GB = 1.0
treatment = sys.argv[1]
assert torch.cuda.is_available(), "需要 CUDA"

# 打印实际导入模块路径（Codex P2：避免旧版模块抢先命中）
import FECNet as _F
import local_remote_grouper_v2 as _v2
print("FECNet.__file__ = %s" % _F.__file__)
print("local_remote_grouper_v2.__file__ = %s" % _v2.__file__)

m = FECNet(num_classes=101, use_remote_g16=True, g16_treatment=treatment).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

# EMA（等价 train_frequency_ncaltech.ModelEMA：shadow = float 深拷贝，
# 每 optimizer.step 后 update；此处内联避免 import 训练脚本的顶层副作用）
class _EMA:
    def __init__(self, net, decay=0.999):
        self.decay = decay
        mod = net.module if hasattr(net, 'module') else net
        self.shadow = {k: v.detach().float().clone() for k, v in mod.state_dict().items()}
    @torch.no_grad()
    def update(self, net):
        mod = net.module if hasattr(net, 'module') else net
        for k, v in mod.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)
ema = _EMA(m, decay=0.999)

torch.cuda.synchronize()
free0, total = torch.cuda.mem_get_info()
reserved0 = torch.cuda.memory_reserved()
# external0 = 非本进程 allocator 的固定占用（其他进程 + CUDA context）
external0 = (total - free0) - reserved0

torch.cuda.reset_peak_memory_stats()
m.train()
for _ in range(N_WINDOWS):
    opt.zero_grad()
    for _ in range(ACCUM):
        x = torch.randn(B, 4, N, device='cuda')
        label = torch.randint(0, 101, (B,), device='cuda')
        out = m(x)
        loss = torch.nn.functional.cross_entropy(out, label)
        (loss / ACCUM).backward()
    opt.step()
    ema.update(m)
    torch.cuda.synchronize()

peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
est_total_used = peak_reserved + external0 / 1024**3
verdict = (est_total_used + SAFETY_MARGIN_GB) < (total / 1024**3)

print("treatment=%s B=%d accum=%d GPU=%s" % (treatment, B, ACCUM, torch.cuda.get_device_name(0)))
print("  allocated_peak=%.2fGB reserved_peak=%.2fGB" % (peak_alloc, peak_reserved))
print("  external0=%.2fGB total=%.2fGB" % (external0 / 1024**3, total / 1024**3))
print("  判定: est_total_used=%.2fGB + margin %.2fGB = %.2fGB vs total %.2fGB -> %s"
      % (est_total_used, SAFETY_MARGIN_GB, est_total_used + SAFETY_MARGIN_GB,
         total / 1024**3, "PASS" if verdict else "FAIL"))
print("  判定表达式: peak_reserved + external0 + %.1fGB < total" % SAFETY_MARGIN_GB)
sys.exit(0 if verdict else 1)
'''

if __name__ == '__main__':
    overall = 0
    for t in TREATMENTS:
        r = subprocess.run([sys.executable, '-c', CHILD, t])
        print("--- %s exit=%d ---" % (t, r.returncode))
        overall = overall or r.returncode
    print("总体判定：%s（任一 treatment 超阈值即 FAIL，exit=%d）"
          % ("PASS" if overall == 0 else "FAIL", overall))
    sys.exit(overall)
