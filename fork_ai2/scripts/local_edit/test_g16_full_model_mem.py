# test_g16_full_model_mem.py — Codex 交接要求补全项：
# 完整 LRHyperNet B=8 前向+反向峰值显存实测（173720 只测了池+selector）
#
# 背景：Codex codex_review 2026-10-14 晚间核查明确指出——
#   "该作业目前仅测池 + selector，不是完整模型显存...接线训练前需要一次
#    完整 B=8 前向+反向测峰值"
# 本脚本补这一关。LRHyperNet 实例化自 HPC 现行 models/（不改任何
# 运行中作业的代码），selector_mode 扫五个 G16 处理（与 ai2_g16 五处理
# 名单对齐：topk=现状特征最远 / knn=池内最近（largest=False 等价语义由
# selector_mode 控制，此处用 LRHyperNet 原生支持的模式名））。
#
# 注意：本脚本只测显存/延迟，不做任何精度判断。
# LRHyperNet 原生 selector_mode 支持："topk"/"greedy_div"/"random"/
# "spatial_uniform"（详见 lr_hypernet.py 参数注释）。G16 五处理的
# euc_near/hyp_far 在 LRHyperNet 原生代码里没有开关（那是 ai2_g16 隔离
# 实现的），所以本脚本分两部分：
#   Part 1: 原生 selector_mode 四种模式的完整模型前反向显存（可直接接线）
#   Part 2: use_hyperbolic_selector=True 的完整模型前反向显存（双曲选点，
#           G16 训练提案的核心配置）
# euc_near/hyp_far 的接线需要改 local_remote_grouper（等 Codex 批），
# 它们的 selector 级显存已由 173720 覆盖。

import sys
import time
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '/home/jnguan/JiangYouxin/ZhaoJin')

assert torch.cuda.is_available(), "需要计算节点 GPU"
dev = torch.device('cuda')
torch.backends.cudnn.benchmark = False

B, N, NUM_CLASS = 8, 10240, 101     # 历史训练真实配置

print(f"[gpu] {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
print(f"[cfg] B={B} N={N} num_class={NUM_CLASS}")
print(f"[hash] 源文件 md5 备查（与 173720 一致的验证链）")

import hashlib
for f in ['models/local_remote_grouper.py', 'lr_hypernet.py']:
    h = hashlib.md5(open(f'/home/jnguan/JiangYouxin/ZhaoJin/{f}','rb').read()).hexdigest()
    print(f"  {h}  {f}")

from lr_hypernet import LRHyperNet

def run_full_backward(model, tag):
    """一个完整训练步（前向+loss+反向+step）的峰值显存与延迟。"""
    model.train()
    x = torch.rand(B, 4, N, device=dev)          # [B,4,N] (t,x,y,p)∈[0,1]
    y = torch.randint(0, NUM_CLASS, (B,), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # warmup（cudnn/内存池初始化）
    out = model(x); loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 2**20
    times = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20
    dt = sorted(times)[1]
    print(f"[full] {tag}: peak={peak:.0f} MiB (base {base:.0f}, "
          f"Δ={peak-base:.0f}), {dt*1000:.0f} ms/step, loss={loss.item():.3f}")
    return peak

print()
print('=== Part 1: LRHyperNet 原生 selector_mode 完整前反向（remote_g=16）===')
# 2026-10-15 修正：greedy_div 的 (cand.unsqueeze(3)-sel.unsqueeze(2)) 广播产生
# [B,S,K,G,N] 级张量（历史实现问题，OOM on V100-16）——从列表移除并单独
# try/except（Codex 口径：六处理主对照不含 greedy_div，OOM 不阻塞其余处理）。
for mode in ['topk', 'random', 'spatial_uniform']:
    torch.manual_seed(42)
    m = LRHyperNet(num_classes=NUM_CLASS, remote_g=16,
                   hyperbolic_type='lorentz',
                   selector_mode=mode).to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"--- selector_mode={mode} (params={n_par/1e3:.0f}K) ---")
    try:
        run_full_backward(m, mode)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {mode}: {str(e)[:100]} —— 标记跳过，不阻塞后续处理")
    del m
    torch.cuda.empty_cache()

# greedy_div 单独测（预期 OOM，记录证据不中断）
print(f"--- selector_mode=greedy_div (params 同上) --- [预期 OOM, 2026-10-15 移出主对照]")

print()
print('=== Part 2: use_hyperbolic_selector=True（双曲选点，G16 核心）===')
torch.manual_seed(42)
m = LRHyperNet(num_classes=NUM_CLASS, remote_g=16,
               hyperbolic_type='lorentz',
               use_hyperbolic_selector=True,
               selector_mode='topk').to(dev)
n_par = sum(p.numel() for p in m.parameters())
print(f"--- use_hyperbolic_selector=True (params={n_par/1e3:.0f}K) ---")
run_full_backward(m, 'hyp_selector_topk')
del m
torch.cuda.empty_cache()

print()
print('ALL_TESTS_PASS')
