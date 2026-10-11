# 验证修正后的 train_hfecnet.py 语法 + 消融模型可导入 + 快速前向
import ast
import sys

# 1. 语法检查
with open('train_hfecnet.py', 'r', encoding='utf-8') as f:
    ast.parse(f.read())
print('[OK] train_hfecnet.py 语法正确')

# 2. 消融模型导入 + 前向
from hyperbolic_fecnet_ablation import HFECNetAblation
import torch

for ablation in ['none', 'no_hyperbolic', 'fixed_curvature', 'no_centroid']:
    m = HFECNetAblation(num_classes=101, ablation=ablation, curvature_init=0.4)
    x = torch.randn(2, 4, 2048)  # [B, 4, N]
    with torch.no_grad():
        out = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f'[OK] ablation={ablation}: out={tuple(out.shape)}, params={params:,}')

print('[ALL PASS]')
