"""G16 真实训练显存测试（3090 上跑，Codex P0-2 要求）。

模拟真实训练：batch=16 + grad_accum=2，跑一个完整 optimizer.step 周期
（2 个 micro-batch 的 forward+backward 梯度累积），记录峰值显存。

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_g16_memory.py
"""
import sys
sys.path.insert(0, '.')
import torch
from FECNet import FECNet

B, N, ACCUM = 16, 10240, 2

m = FECNet(num_classes=101, use_remote_g16=True, g16_treatment='hyp_far').cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

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
peak = torch.cuda.max_memory_allocated() / 1024**3
print('G16 batch%d grad_accum%d 峰值显存 = %.2f GB' % (B, ACCUM, peak))
print('判定：< 24GB 可跑训练；> 24GB 需降 batch')
