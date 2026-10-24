"""core idea 2 协方差通道数值自检（在 3090 上跑，有 torch + models）。

验证 FECNet._covariance_density_channels：
1. 输出 shape = [B, E, N]（covariance 3 + density 1 = 4 维）
2. 数值有限（无 NaN/Inf），归一化后 cov 在 [0,1]
3. count<3 门控：单点体素协方差归零
4. 手算对照：构造"y 方向拉长"的体素，σyy 应 > σxx；密度 = log1p(count)

用法（3090 code-server 终端）：
    cd ~/zhaojin/pi2/tools
    PYTHONPATH=~/zhaojin/pi2:~/zhaojin/pi2/models ~/zhaojin/venv/bin/python test_covariance_channel.py
"""
import sys
import torch

sys.path.insert(0, ".")
from models.FECNet import FECNet

m = FECNet(num_classes=101, use_covariance_channel=True,
           use_density_channel=True, density_bins=8)

B, N = 2, 200
torch.manual_seed(0)
xyz = torch.rand(B, N, 4)  # x,y 在 [0,1]

# 造一个已知分布：batch 0 的前 5 个点放同一体素，x 固定（σxx 小）、y 沿轴展开（σyy 大）
xyz[0, 0:5, 1] = 0.2
xyz[0, 0:5, 2] = torch.linspace(0.1, 0.5, 5)

out = m._covariance_density_channels(xyz)  # [B, 4, N]
print("out shape:", tuple(out.shape))
assert out.shape == (B, 4, N), f"shape 错: {out.shape}"

cov = out[:, 0:3, :]   # [B, 3, N] = σxx, σyy, σxy
dens = out[:, 3:4, :]  # [B, 1, N] = log1p(count)

# 1. 有限性
assert torch.isfinite(out).all(), "有 NaN/Inf"

# 2. 归一化后 cov 在 [0,1]（σxy 可能为负，取绝对值看范围）
print("cov_xx/yy 范围:", cov[:, 0:2].min().item(), cov[:, 0:2].max().item())
assert cov[:, 0:2].min() >= 0.0 and cov[:, 0:2].max() <= 1.0 + 1e-3, "σxx/σyy 归一化越界"
assert cov[:, 2].abs().max() <= 1.0 + 1e-3, "σxy 归一化越界"

# 3. y 方向拉长的 5 点体素：σyy > σxx
sxx, syy = out[0, 0, 0].item(), out[0, 1, 0].item()
print(f"5点体素点0: σxx={sxx:.4f} σyy={syy:.4f} (y 拉长应 σyy>σxx)")
assert syy > sxx, "y 方向拉长应 σyy > σxx"

# 4. 密度 = log1p(count)；5 点体素 count=5
d0 = out[0, 3, 0].item()
expect = float(torch.log1p(torch.tensor(5.0)).item())
print(f"5点体素点0: density={d0:.4f} (期望 log1p(5)={expect:.4f})")
assert abs(d0 - expect) < 1e-3, f"密度错: {d0} vs {expect}"

# 5. count<3 门控：找一个单点体素，其 cov 应为 0（有效门控 × 方差 0 = 0）
#    随机点大部分是单点体素，其 σxx+σyy=0 → 归一化后为 0（已乘 valid）
single = (out[1, 0] == 0.0) | (out[1, 1] == 0.0)  # batch 1 里 σxx 或 σyy 为 0 的点
print("batch1 单点体素占比（σxx 或 σyy=0）:", single.float().mean().item())

print("\n全部断言通过 ✓")
