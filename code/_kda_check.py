"""KDA 因果编码器数值正确性验证：einsum 实现 vs 逐元素暴力实现。

验证 gated delta rule 递推每一步与暴力循环完全一致（含排序、状态更新、读出）。
"""
import torch
import torch.nn as nn
from models.local_remote_grouper import KDACausalEncoder


def brute_force_kda(enc: KDACausalEncoder, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """完全按公式逐元素实现，不依赖任何张量化 trick。"""
    B, S, K, C = x.shape
    D = enc.state_dim
    # 排序（与 enc 内部一致）
    order = ts[..., 0].argsort(dim=-1)  # [B,S,K]
    x_sorted = torch.gather(x, 2, order.unsqueeze(-1).expand_as(x))
    with torch.no_grad():
        k = enc.k_proj(x_sorted)
        v = enc.v_proj(x_sorted)
        q = enc.q_proj(x_sorted)
        beta = torch.sigmoid(enc.beta_proj(x_sorted))
        alpha = torch.exp(enc.alpha_proj(x_sorted)).clamp(max=1.0)
    outs = torch.zeros(B, S, K, D)
    for b in range(B):
        for s in range(S):
            state = torch.zeros(D, D)
            for i in range(K):
                ki = k[b, s, i]              # [D]
                vi = v[b, s, i]
                qi = q[b, s, i]
                bi = beta[b, s, i].item()
                ai = alpha[b, s, i]          # [D]
                # S <- Diag(alpha) S - beta k (k^T S) + beta k v^T
                kS = ki @ state               # [D]
                state = ai.unsqueeze(1) * state - bi * torch.outer(ki, kS) + bi * torch.outer(ki, vi)
                outs[b, s, i] = state.T @ qi
    # out_proj + layer_norm（与 enc 相同）
    return enc.layer_norm(enc.out_proj(outs))


def main():
    torch.manual_seed(0)
    for (B, S, K, C) in [(2, 8, 24, 64), (3, 5, 13, 32), (1, 4, 7, 16)]:
        enc = KDACausalEncoder(C)
        enc.eval()
        x = torch.randn(B, S, K, C)
        ts = torch.rand(B, S, K, 1)  # 时间戳 [0,1)
        with torch.no_grad():
            out_fast = enc(x, ts)
            out_brute = brute_force_kda(enc, x, ts)
        diff = (out_fast - out_brute).abs().max().item()
        rel = diff / out_brute.abs().max().item()
        print(f"[B,S,K,C]={list((B,S,K,C))}: max_abs_diff={diff:.2e}, rel={rel:.2e}")
        assert diff < 1e-5, "数值不一致！"
    print("KDA 递推正确性验证通过 ✅")

    # 因果性验证：组内事件顺序颠倒（x 与 ts 一起翻转，保持绑定）时，
    # 输出必须完全一致——因为内部会重新按时间升序排列
    enc = KDACausalEncoder(64)
    enc.eval()
    x = torch.randn(2, 8, 24, 64)
    ts = torch.rand(2, 8, 24, 1)
    with torch.no_grad():
        out1 = enc(x, ts)
        out2 = enc(x.flip(-2), ts.flip(-2))
    print(f"时间顺序不变性 max_diff={ (out1 - out2).abs().max().item():.2e}（应≈0）")
    assert (out1 - out2).abs().max().item() < 1e-5, "时间排序处理不一致！"

    # 常数时间戳（全 0）：排序退化为原始顺序，仍应有限
    with torch.no_grad():
        out3 = enc(x, torch.zeros_like(ts))
    print(f"常数时间戳输出 finite={torch.isfinite(out3).all().item()}")
    assert torch.isfinite(out3).all().item()

    print("全部 KDA 数值验证通过 ✅✅✅")


if __name__ == "__main__":
    main()
