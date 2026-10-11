"""
CPU 前向传播测试：验证 FECNet 能正常跑通
"""
import sys
import os
import torch

# 把 models 目录加入 path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

from models.FECNet import FECNet

print("=== 创建模型 ===")
model = FECNet(num_classes=101)
model.eval()

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

print("\n=== 前向传播测试 ===")
# 输入: [B, 4, N]，4 是 (x,y,t,p)，N=8192 点
batch_size = 2
num_points = 8192
x = torch.randn(batch_size, 4, num_points)
print(f"Input shape: {x.shape}")

with torch.no_grad():
    try:
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output sample (first 5): {output[0, :5]}")
        print("\n✅ 前向传播成功！")
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

print("\n=== 反向传播测试 ===")
model.train()
x = torch.randn(batch_size, 4, num_points)
label = torch.randint(0, 101, (batch_size,))
try:
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, label)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print("✅ 反向传播成功！")
except Exception as e:
    print(f"❌ 反向传播失败: {e}")
    import traceback
    traceback.print_exc()
