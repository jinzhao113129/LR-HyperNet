"""
v3 (KDA) 显存实测脚本：batch=80 前向+反向，打印 CUDA 峰值显存。

Claude 审查建议：KDA state [B,S,16,16] 本身 ~300MB，但 K/V/Q 投影、outs
列表与计算图可能额外占 1.5-2GB。用 torch.cuda.max_memory_allocated() 实测
batch=80 的峰值，确认 A100/H100 80GB 上无 OOM 风险。

用法（在 A100/H100 作业内）：
    python _v3_mem_check.py
"""
import torch
from lr_hypernet import LRHyperNet


def main():
    torch.manual_seed(0)
    device = 'cuda'
    model = LRHyperNet(num_classes=101, kda=True, use_global_anchor=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"v3 params: {total_params:,}")
    x = torch.randn(80, 4, 10240, device=device)
    label = torch.randint(0, 101, (80,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, label)
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step}: non-finite loss {loss.item()}")
        loss.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"step {step}: loss={loss.item():.4f}, peak_mem={peak:.1f} MB", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"MAX_MEM_ALLOCATED={peak:.1f} MB ({(peak/1024):.2f} GB)")
    assert torch.isfinite(loss)
    print("v3 mem check PASS")


if __name__ == "__main__":
    main()
