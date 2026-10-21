"""G16 真实训练显存测试（3090 上跑，Codex P0-2/P1-4 要求）。

模拟真实训练：batch=16 + grad_accum=2，跑一个完整 optimizer.step 周期
（2 个 micro-batch 的 forward+backward 梯度累积），记录 allocated/reserved/可用显存。
测 hyp_far（双曲远程选择，学长核心）和 rp1_scorer（RP1 空间新颖性，可能更重）
两个 treatment，rp1_scorer 作为最大内存代表。

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_g16_memory.py
"""
import sys
sys.path.insert(0, '.')
import torch
from FECNet import FECNet

B, N, ACCUM = 16, 10240, 2


def mem_report(tag):
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    free, total = torch.cuda.mem_get_info()
    print('%s: allocated=%.2fGB reserved=%.2fGB 可用=%.2fGB/%.2fGB'
          % (tag, alloc, reserved, free / 1024**3, total / 1024**3))


for treatment in ['hyp_far', 'rp1_scorer']:
    m = FECNet(num_classes=101, use_remote_g16=True,
               g16_treatment=treatment).cuda()
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
    mem_report('treatment=%s' % treatment)

print('判定：allocated/reserved + 安全余量 < 24GB 才可放行训练')
