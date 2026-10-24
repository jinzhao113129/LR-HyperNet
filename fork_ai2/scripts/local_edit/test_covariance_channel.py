"""core idea 2 协方差通道数值自检（在 3090 上跑，有 torch + models）。

验证 FECNet._covariance_density_channels 的核心逻辑（用干净的同体素数据，避免
随机点混入）：
1. 输出 shape = [B, E, N]（covariance 3 + density 1 = 4 维）
2. 5 点同体素、x 固定 y 展开 → σxx≈0、σyy>0、σxy≈0（手算对照）
3. 密度 = log1p(count)
4. count<3 门控：单点体素协方差归零

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_covariance_channel.py
"""
import sys
import torch

sys.path.insert(0, ".")
from models.FECNet import FECNet

# bins=2 → 体素宽 0.5；5 点全落进同一体素（x=0.2→列0，y 均<0.5→行0）
m = FECNet(num_classes=101, use_covariance_channel=True,
           use_density_channel=True, density_bins=2)

B, N = 1, 5
xyz = torch.zeros(B, N, 4)
xyz[0, :, 0] = 0.5   # t 任意
xyz[0, :, 1] = 0.2   # x 固定 → σxx≈0
xyz[0, :, 2] = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.45])  # y 展开（<0.5 同行）→ σyy>0
xyz[0, :, 3] = 1.0   # p 任意

out = m._covariance_density_channels(xyz)  # [B, 4, N]
print("out shape:", tuple(out.shape))
assert out.shape == (B, 4, N), f"shape 错: {out.shape}"

sxx, syy, sxy = out[0, 0, 0].item(), out[0, 1, 0].item(), out[0, 2, 0].item()
dens = out[0, 3, 0].item()

print(f"σxx={sxx:.5f}（期望≈0，x 固定）")
print(f"σyy={syy:.5f}（期望>0，y 展开）")
print(f"σxy={sxy:.5f}（期望≈0，x 固定无相关）")
print(f"density={dens:.5f}（期望 log1p(5)={float(torch.log1p(torch.tensor(5.0))):.5f}）")

assert torch.isfinite(out).all(), "有 NaN/Inf"
assert abs(sxx) < 1e-5, f"σxx 应≈0，实际 {sxx}"
assert syy > 0, f"σyy 应>0，实际 {syy}"
assert syy > sxx, "σyy 应 > σxx"
assert abs(sxy) < 1e-5, f"σxy 应≈0，实际 {sxy}"
assert abs(dens - float(torch.log1p(torch.tensor(5.0)))) < 1e-3, f"密度错: {dens}"

# count<3 门控：单点体素（N=1）协方差应归零
m2 = FECNet(num_classes=101, use_covariance_channel=True,
            use_density_channel=False, density_bins=4)
xyz1 = torch.zeros(1, 1, 4)
xyz1[0, 0, 1] = 0.3
xyz1[0, 0, 2] = 0.4
out1 = m2._covariance_density_channels(xyz1)  # [1, 3, 1]
print(f"单点体素 cov: {out1[0, :, 0].tolist()}（期望全 0，count<3 门控）")
assert out1[0, :, 0].abs().max() < 1e-6, "单点体素协方差应归零"

print("\n全部断言通过 ✓")
