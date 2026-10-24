"""learned scorer（core idea 1）自检：构造 learned FECNet，跑 forward + backward 验证。

验证：
1. learned 分支 forward 不报错、输出 finite、shape 对
2. 每个 stage 的 learned_scorer 存在、有参数
3. backward 后 scorer 梯度非零（可微选点，梯度回流）

用法（3090）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_learned_scorer.py
"""
import sys
import torch

sys.path.insert(0, ".")
from models.FECNet import FECNet

m = FECNet(num_classes=101, use_remote_g16=True, g16_treatment='learned').cuda()

B, N = 2, 10240
torch.manual_seed(0)

# 1. eval 前向：不报错、finite、shape
m.eval()
with torch.no_grad():
    out = m(torch.randn(B, 4, N).cuda())
print("eval out shape:", tuple(out.shape))
assert out.shape == (B, 101), out.shape
assert torch.isfinite(out).all(), "eval 输出有 NaN/Inf"

# 2. 每个 stage 的 learned_scorer 存在
for i, g in enumerate(m.local_grouper_list):
    sel = g.remote_selector
    assert sel.g16_treatment == 'learned', f"stage {i} treatment={sel.g16_treatment}"
    assert hasattr(sel, 'learned_scorer'), f"stage {i} 缺 learned_scorer"
    n = sum(p.numel() for p in sel.learned_scorer.parameters())
    print(f"stage {i}: learned_scorer 参数 {n}")

# 3. train 前向 + backward：scorer 梯度非零（可微）
m.train()
x = torch.randn(B, 4, N).cuda()
out = m(x)
loss = out.mean()
loss.backward()
for i, g in enumerate(m.local_grouper_list):
    sel = g.remote_selector
    grads = [p.grad for p in sel.learned_scorer.parameters()]
    ok = all(grad is not None and grad.abs().sum().item() > 0 for grad in grads)
    print(f"stage {i}: scorer 梯度非零 = {ok}")
    assert ok, f"stage {i} scorer 梯度为零（不可微）"

print("\nlearned scorer 自检通过 ✓（forward 跑通 + 梯度回流）")
