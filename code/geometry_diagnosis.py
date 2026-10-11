"""
几何诊断脚本：判断双曲距离是否真正改变了注意力结构
原理：如果 d_hyp/d_euc 比值近似常数，softmax(-d_hyp/T) 与 softmax(-d_euc²/T)
      产生的注意力权重几乎相同 → 双曲和欧氏不可能有差异 → 可以直接判死
用法：加载训练好的模型，对一批测试数据记录 remote_points 范数分布和 d_hyp/d_euc 比值
"""
import torch
import numpy as np
import h5py
import sys
sys.path.insert(0, '/home/jnguan/JiangYouxin/ZhaoJin')

from models.local_remote_grouper import LocalRemoteGrouper
from lr_hypernet import LRHyperNet


def load_data(data_path, num_point=10240, num_samples=50):
    """加载测试数据的前 num_samples 个样本"""
    with h5py.File(f'{data_path}/test.h5', 'r') as f:
        data = f['data'][:num_samples]  # [N, 10240, 4]
        label = f['label'][:num_samples]
    data = data.reshape(-1, num_point, 4).astype(np.float32)
    return torch.from_numpy(data), torch.from_numpy(label).long()


def compute_distances(points, anchor, hyperbolic_type='lorentz', curvature=0.4):
    """
    计算一批点到锚点的双曲距离和欧氏距离
    points: [B, K, D] 远程点特征
    anchor: [B, 1, D] 锚点特征
    返回: d_hyp [B, K], d_euc [B, K]
    """
    # 欧氏距离
    d_euc = torch.norm(points - anchor, dim=-1)  # [B, K]

    # 双曲距离（Lorentz模型）
    if hyperbolic_type == 'lorentz':
        c = curvature
        # Lorentz 内积: <x,y>_L = x0*y0 - sum(xi*yi)
        # 假设 points 和 anchor 已经在 Lorentz 流形上（x0 = sqrt(1/c + ||x_rest||²)）
        x0 = points[..., 0:1]
        x_rest = points[..., 1:]
        a0 = anchor[..., 0:1]
        a_rest = anchor[..., 1:]
        inner = x0 * a0 - torch.sum(x_rest * a_rest, dim=-1, keepdim=True)
        inner = inner.clamp(min=1.0 + 1e-6)  # 保证 >=1
        d_hyp = (1.0 / np.sqrt(c)) * torch.acosh(1.0 + 2 * c * (inner - 1.0))
        d_hyp = d_hyp.squeeze(-1)
    elif hyperbolic_type == 'poincare':
        c = curvature
        # Poincaré 球模型距离
        norm_p = torch.norm(points, dim=-1)
        norm_a = torch.norm(anchor, dim=-1)
        diff_norm = torch.norm(points - anchor, dim=-1)
        denom = (1 - c * norm_p**2) * (1 - c * norm_a**2)
        denom = denom.clamp(min=1e-6)
        cosh_arg = 1 + 2 * c * diff_norm**2 / denom
        cosh_arg = cosh_arg.clamp(min=1.0 + 1e-6)
        d_hyp = (1.0 / np.sqrt(c)) * torch.acosh(cosh_arg)
    else:
        d_hyp = d_euc  # euclidean  fallback

    return d_hyp, d_euc


def diagnose(model, data, device='cpu', num_batches=5):
    """
    对模型的每个 stage 记录：
    1. remote_points 范数分布
    2. d_hyp/d_euc 比值分布
    3. 双曲注意力权重 vs 欧氏注意力权重的 KL 散度
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)

    results = {}

    # hook 捕获每个 stage 的 remote_points 和 anchor
    stage_data = []

    def make_hook(stage_idx):
        def hook(module, input, output):
            # output 是聚合后的特征，我们需要从 module 内部取中间值
            # 这里简化：直接记录输入特征的统计
            pass
        return hook

    # 简化版：直接用模型的 grouper 计算距离
    with torch.no_grad():
        # 前向传播，同时收集统计
        x = data
        for i, stage in enumerate(model.stages):
            # 获取该 stage 的输入
            B, N, D = x.shape
            # 局部分组
            local_xyz, local_points, idx = stage.grouping(x[:, :, :4], x)
            # 远程选择和聚合
            remote_idx = stage.remote_selector(local_points, idx)
            remote_points = stage.index_points(x, remote_idx)  # [B, S, K, D]
            anchor = remote_points.mean(dim=2, keepdim=True)  # [B, S, 1, D]

            # 计算距离
            B_s, S, K, D_s = remote_points.shape
            rp_flat = remote_points.reshape(B_s * S, K, D_s)
            a_flat = anchor.reshape(B_s * S, 1, D_s)

            d_hyp, d_euc = compute_distances(
                rp_flat, a_flat,
                hyperbolic_type=stage.hyper_agg.hyperbolic_type
                if hasattr(stage.hyper_agg, 'hyperbolic_type') else 'lorentz',
                curvature=stage.hyper_agg.curvature.item()
                if hasattr(stage.hyper_agg, 'curvature') else 0.4
            )

            ratio = (d_hyp / (d_euc + 1e-8)).cpu().numpy()
            rp_norm = torch.norm(rp_flat, dim=-1).cpu().numpy()

            # 注意力权重对比
            T_hyp = stage.hyper_agg.curvature.item() if hasattr(stage.hyper_agg, 'curvature') else 0.4
            T_euc = 1.0
            attn_hyp = torch.softmax(-d_hyp / (T_hyp + 1e-8), dim=-1)
            attn_euc = torch.softmax(-d_euc**2 / (T_euc + 1e-8), dim=-1)
            kl = torch.sum(attn_hyp * (torch.log(attn_hyp + 1e-8) - torch.log(attn_euc + 1e-8)), dim=-1)
            kl = kl.cpu().numpy()

            stage_data.append({
                'stage': i,
                'ratio_mean': ratio.mean(),
                'ratio_std': ratio.std(),
                'ratio_min': ratio.min(),
                'ratio_max': ratio.max(),
                'ratio_cv': ratio.std() / (ratio.mean() + 1e-8),  # 变异系数
                'rp_norm_mean': rp_norm.mean(),
                'rp_norm_std': rp_norm.std(),
                'kl_mean': kl.mean(),
                'kl_max': kl.max(),
            })

            x = stage(x)

    return stage_data


def print_diagnosis(stage_data):
    """打印诊断结果，给出判死建议"""
    print("=" * 70)
    print("几何诊断结果")
    print("=" * 70)
    print(f"{'Stage':<8} {'ratio_mean':<12} {'ratio_cv':<12} {'ratio_range':<20} {'KL_mean':<10} {'KL_max':<10}")
    print("-" * 70)

    all_cv = []
    all_kl = []
    for d in stage_data:
        ratio_range = f"[{d['ratio_min']:.3f}, {d['ratio_max']:.3f}]"
        print(f"{d['stage']:<8} {d['ratio_mean']:<12.4f} {d['ratio_cv']:<12.4f} {ratio_range:<20} {d['kl_mean']:<10.6f} {d['kl_max']:<10.6f}")
        all_cv.append(d['ratio_cv'])
        all_kl.append(d['kl_mean'])

    print("=" * 70)
    avg_cv = np.mean(all_cv)
    avg_kl = np.mean(all_kl)
    print(f"平均变异系数 CV = {avg_cv:.4f}")
    print(f"平均 KL 散度 = {avg_kl:.6f}")
    print()

    # 判死规则
    if avg_cv < 0.05 and avg_kl < 0.001:
        print("【判死】d_hyp/d_euc 比值近似常数（CV<5%），注意力权重几乎相同（KL<0.001）")
        print("       → 双曲和欧氏不可能有差异，无论跑多少ep都一样")
        print("       → 建议：直接砍掉双曲，不用等30ep对照结果")
    elif avg_cv < 0.15:
        print("【弱信号】比值变化较小（CV<15%），双曲对注意力结构的改变有限")
        print("         → 预期双曲增益很小（<1pp），建议跑完对照确认")
    else:
        print("【有信号】比值变化较大（CV≥15%），双曲真正改变了注意力结构")
        print("         → 值得等完整对照结果，双曲可能有真实增益")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data_path', type=str, default='./ncaltech101_official_baidu/')
    parser.add_argument('--num_point', type=int, default=10240)
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 从 checkpoint 恢复模型配置（简化：用默认配置）
    model = LRHyperNet(
        num_category=101,
        input_dim=4,
        embed_dim=64,
        feature_list=[64, 132, 268, 540],
        group_number=[2048, 1024, 512],
        num_classes=101,
        hyperbolic_type='lorentz',
        time_enc=True,
    )
    model.load_state_dict(checkpoint['net'], strict=False)
    model.eval()

    # 加载数据
    data, label = load_data(args.data_path, args.num_point, args.num_samples)
    print(f"加载 {args.num_samples} 个测试样本，数据形状: {data.shape}")

    # 诊断
    stage_data = diagnose(model, data, device=device)
    print_diagnosis(stage_data)
