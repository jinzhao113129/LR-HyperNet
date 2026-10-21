"""G16 真实训练显存测试（3090 上跑，Codex P0-2/P1-4/P2 要求）。

每个 treatment 放进**独立子进程**（避免上一轮 CUDA 计算图残留污染下一轮峰值，
否则 rp1_scorer 会被 hyp_far 的激活残留挤到假 OOM/虚高峰值）。

记录：baseline（模型+Adam 状态+CUDA context，reset 前）、allocated、reserved、
可用显存、GPU 名称。判定用 reserved + baseline + 安全余量 < 24GB，而非只看瞬态 allocated。

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_g16_memory.py
"""
import sys
import subprocess

B, N, ACCUM = 16, 10240, 2
TREATMENTS = ['hyp_far', 'rp1_scorer']

CHILD = r'''
import sys, torch
sys.path.insert(0, '.')
from FECNet import FECNet
B, N, ACCUM = 16, 10240, 2
treatment = sys.argv[1]
assert torch.cuda.is_available(), "需要 CUDA"
print("GPU: %s" % torch.cuda.get_device_name(0))
m = FECNet(num_classes=101, use_remote_g16=True, g16_treatment=treatment).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
torch.cuda.synchronize()
free0, total = torch.cuda.mem_get_info()
baseline = (total - free0) / 1024**3   # 模型参数 + Adam 状态 + CUDA context
torch.cuda.reset_peak_memory_stats()
m.train()
for step in range(ACCUM):
    x = torch.randn(B, 4, N, device='cuda')
    label = torch.randint(0, 101, (B,), device='cuda')
    if step % ACCUM == 0:
        opt.zero_grad()
    out = m(x)
    loss = torch.nn.functional.cross_entropy(out, label)
    (loss / ACCUM).backward()
    if (step + 1) % ACCUM == 0:
        opt.step()
alloc = torch.cuda.max_memory_allocated() / 1024**3
reserved = torch.cuda.max_memory_reserved() / 1024**3
torch.cuda.synchronize()
free1, _ = torch.cuda.mem_get_info()
print("treatment=%s baseline=%.2fGB allocated=%.2fGB reserved=%.2fGB 可用=%.2fGB/%.2fGB"
      % (treatment, baseline, alloc, reserved, free1/1024**3, total/1024**3))
'''

if __name__ == '__main__':
    for t in TREATMENTS:
        subprocess.run([sys.executable, '-c', CHILD, t], check=True)
    print('判定：reserved + baseline + 安全余量 < 24GB 才可放行训练')
