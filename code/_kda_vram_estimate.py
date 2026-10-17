"""KDA v3 显存精确估算：CPU上跑前向+反向，统计所有中间张量大小。"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

from lr_hypernet import LRHyperNet

def tensor_bytes(t):
    """计算张量占用字节数（含梯度）。"""
    b = t.numel() * t.element_size()
    if t.requires_grad and t.grad is not None:
        b += t.grad.numel() * t.grad.element_size()
    return b

def estimate_vram(batch_size=80, num_points=10240, kda=True):
    """估算batch_size下的峰值显存（参数+激活+梯度+优化器状态）。"""
    torch.manual_seed(42)
    model = LRHyperNet(num_classes=101, kda=kda, use_global_anchor=True)
    model.train()

    # 1. 参数显存
    param_bytes = 0
    for p in model.parameters():
        param_bytes += p.numel() * p.element_size()
        if p.requires_grad:
            param_bytes += p.numel() * p.element_size()  # 梯度
            param_bytes += 2 * p.numel() * 4  # AdamW: m + v (float32)

    # 2. 前向激活显存（用hook统计）
    activation_bytes = [0]
    peak_activation = [0]

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_bytes[0] += output.numel() * output.element_size()
            peak_activation[0] = max(peak_activation[0], activation_bytes[0])
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_bytes[0] += o.numel() * o.element_size()

    handles = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
            handles.append(m.register_forward_hook(hook))

    # 3. KDA内部state显存（单独统计，因为在循环中）
    kda_state_bytes = 0
    kda_intermediate_bytes = 0
    # stage1: S=2048, stage2: S=1024, stage3: S=512
    for S in [2048, 1024, 512]:
        D = 16
        K = 24
        # state [B,S,D,D]
        kda_state_bytes += batch_size * S * D * D * 4
        # k,v,q [B,S,K,D] each
        kda_intermediate_bytes += 3 * batch_size * S * K * D * 4
        # beta [B,S,K,1], alpha [B,S,K,D]
        kda_intermediate_bytes += batch_size * S * K * 1 * 4
        kda_intermediate_bytes += batch_size * S * K * D * 4
        # outs list: K * [B,S,D]
        kda_intermediate_bytes += K * batch_size * S * D * 4
        # kS intermediate [B,S,D] per step (saved for backward)
        kda_intermediate_bytes += K * batch_size * S * D * 4

    # 4. 跑一次前向+反向统计实际激活
    x = torch.randn(batch_size, 4, num_points)
    label = torch.randint(0, 101, (batch_size,))

    # 前向
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, label)

    # 统计计算图中保存的张量（粗略：所有requires_grad的中间张量）
    # 用反向传播前的内存快照近似
    loss.backward()

    for h in handles:
        h.remove()

    # 5. 输入/标签显存
    input_bytes = batch_size * 4 * num_points * 4
    label_bytes = batch_size * 8  # long

    total = param_bytes + activation_bytes[0] + kda_state_bytes + kda_intermediate_bytes + input_bytes + label_bytes

    print(f"\n{'='*60}")
    print(f"KDA v3 显存估算 (batch={batch_size}, N={num_points})")
    print(f"{'='*60}")
    print(f"参数+梯度+AdamW状态: {param_bytes/1024**2:.1f} MB")
    print(f"前向激活(hook统计):   {activation_bytes[0]/1024**2:.1f} MB")
    print(f"KDA state(3stage):    {kda_state_bytes/1024**2:.1f} MB")
    print(f"KDA中间张量(k/v/q等): {kda_intermediate_bytes/1024**2:.1f} MB")
    print(f"输入+标签:            {(input_bytes+label_bytes)/1024**2:.1f} MB")
    print(f"{'-'*60}")
    print(f"估算总峰值:           {total/1024**2:.1f} MB ({total/1024**3:.2f} GB)")
    print(f"{'='*60}")

    # 对比无KDA版本
    print(f"\n对比: 无KDA(v1)估算")
    model_v1 = LRHyperNet(num_classes=101, kda=False, use_global_anchor=True)
    param_v1 = sum(p.numel() * p.element_size() * 4 for p in model_v1.parameters())  # param+grad+2*adam
    print(f"  参数+梯度+AdamW: {param_v1/1024**2:.1f} MB")
    print(f"  KDA额外开销:     {(param_bytes - param_v1 + kda_state_bytes + kda_intermediate_bytes)/1024**2:.1f} MB")

    return total

if __name__ == "__main__":
    for bs in [16, 32, 64, 80]:
        try:
            estimate_vram(batch_size=bs)
        except Exception as e:
            print(f"batch={bs} failed: {e}")
            import traceback
            traceback.print_exc()
