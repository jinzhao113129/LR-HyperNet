# -*- coding: utf-8 -*-
"""
HFECNet 消融实验版本

基于 hyperbolic_fecnet.py，通过 ablation 参数切换 4 种组内聚合方式：
  --ablation none            : 完整双曲模型（双曲距离注意力 + 质心锚点 + 可学习曲率）
  --ablation no_hyperbolic   : 去掉双曲聚合，用普通欧氏注意力加权和（聚合步退化为 FECNet）
  --ablation fixed_curvature : 曲率固定为 1.0（不可学习，其余双曲部分保留）
  --ablation no_centroid     : 不用组内质心做锚点，改用组内第一个点做锚点

设计原则：继承原模型结构，只改聚合部分（HyperbolicGroupAggregator 的子类），
模型主体 HFECNetAblation 直接继承 HFECNet，仅按消融模式重建聚合层列表，
其余（LocalGrouper / 空间FFT / 时间FFT / classifier）与原模型完全一致，
保证各消融版本之间参数量与结构可比。
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_fecnet import HyperbolicGroupAggregator, HFECNet


# ============================================================================
# 消融聚合模块（均继承自 HyperbolicGroupAggregator，只改聚合逻辑）
# ============================================================================

class FullHyperbolicAggregator(HyperbolicGroupAggregator):
    """消融 'none'：完整双曲聚合，不做任何修改，直接继承。"""


class EuclideanAggregator(HyperbolicGroupAggregator):
    """消融 'no_hyperbolic'：去掉双曲几何，用欧氏注意力加权和。

    聚合步与原版 FECNet 完全一致：
      线性打分 (Attention) → softmax 得到权重 → bmm 加权和
    为了让消融只隔离"双曲 vs 欧氏"这一个变量，
    输出头（out_proj + 残差 + LayerNorm）与其他消融版本保持相同。
    双曲曲率参数在此版本中无意义，直接删除。
    """

    def __init__(self, hidden_size: int, curvature_init: float = 0.4,
                 temperature: float = 1.0, use_centroid: bool = True):
        super().__init__(hidden_size, curvature_init, temperature, use_centroid)
        # 曲率参数在欧氏版本中不参与计算，删除以免干扰参数统计
        del self.curvature
        # 欧氏注意力打分器（对应 FECNet 的 Attention：Linear(C -> 1)）
        self.attn_linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*G, K, C]，K 个邻居点的欧氏特征
        Returns:
            [B*G, C]，聚合后的欧氏特征
        """
        # 1. 欧氏注意力打分（FECNet 式：Linear + softmax，沿 K 维归一化）
        scores = self.attn_linear(x).squeeze(-1)          # [B*G, K]
        attn_weights = F.softmax(scores, dim=-1)          # [B*G, K]

        # 2. 欧氏空间注意力加权和（与 FECNet 的 bmm 聚合一致）
        agg_euclidean = torch.bmm(
            attn_weights.unsqueeze(1), x).squeeze(1)      # [B*G, C]

        # 3. 输出投影 + 残差 + LayerNorm（与其他消融版本保持一致）
        out = self.out_proj(agg_euclidean)
        residual = x.mean(dim=1)                          # [B*G, C]
        out = self.layer_norm(out + residual)
        return out


class FixedCurvatureAggregator(HyperbolicGroupAggregator):
    """消融 'fixed_curvature'：曲率固定为 1.0，不可学习。

    把可学习的曲率 nn.Parameter 替换为固定 buffer，
    其余双曲流程（指数映射 / 双曲距离注意力 / 切空间聚合）全部保留，
    用于验证"曲率自适应学习"是否带来增益。
    """

    def __init__(self, hidden_size: int, curvature_init: float = 0.4,
                 temperature: float = 1.0, use_centroid: bool = True,
                 fixed_curvature: float = 1.0):
        super().__init__(hidden_size, curvature_init, temperature, use_centroid)
        # 删除可学习曲率参数，换成固定 buffer（不注册为 Parameter，不参与优化）
        del self.curvature
        self.register_buffer('curvature_fixed',
                             torch.tensor(float(fixed_curvature)))

    def _curvature(self) -> torch.Tensor:
        """固定曲率：直接返回常量 1.0。"""
        return self.curvature_fixed


class FirstPointAnchorAggregator(HyperbolicGroupAggregator):
    """消融 'no_centroid'：不用组内质心做锚点，用组内第一个点做锚点。

    直接继承 HyperbolicGroupAggregator 并令 use_centroid=False，
    锚点从"组内欧氏质心映射到双曲空间"变为"组内第一个点（已映射）"，
    用于验证"质心锚点"这一设计的必要性。
    """

    def __init__(self, hidden_size: int, curvature_init: float = 0.4,
                 temperature: float = 1.0):
        super().__init__(hidden_size, curvature_init, temperature,
                         use_centroid=False)


# 消融模式 -> 聚合器构造函数 的注册表
def build_aggregator(ablation: str, hidden_size: int, curvature_init: float):
    """根据消融模式构造对应的组内聚合模块。"""
    if ablation == 'none':
        return FullHyperbolicAggregator(
            hidden_size, curvature_init=curvature_init, use_centroid=True)
    elif ablation == 'no_hyperbolic':
        return EuclideanAggregator(
            hidden_size, curvature_init=curvature_init, use_centroid=True)
    elif ablation == 'fixed_curvature':
        return FixedCurvatureAggregator(
            hidden_size, curvature_init=curvature_init, use_centroid=True,
            fixed_curvature=1.0)
    elif ablation == 'no_centroid':
        return FirstPointAnchorAggregator(
            hidden_size, curvature_init=curvature_init)
    else:
        raise ValueError(
            f"未知消融模式: {ablation}，可选 none / no_hyperbolic / "
            f"fixed_curvature / no_centroid")


# ============================================================================
# 消融版 HFECNet：继承原模型，只重建聚合层
# ============================================================================

class HFECNetAblation(HFECNet):
    """HFECNet 消融版。

    继承 hyperbolic_fecnet.HFECNet 的全部结构，
    仅按 ablation 模式重建 hyp_aggregation_list（组内聚合层），
    其余层（embed_dim / LocalGrouper / FFT / classifier）与原模型完全相同。
    """

    VALID_ABLATIONS = ('none', 'no_hyperbolic', 'fixed_curvature', 'no_centroid')

    def __init__(self, num_classes: int, ablation: str = 'none',
                 curvature_init: float = 0.4):
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"未知消融模式: {ablation}，可选 {self.VALID_ABLATIONS}")
        # 先按完整双曲模型构建（use_centroid=True），再按消融模式重建聚合层
        super().__init__(num_classes, curvature_init=curvature_init,
                         use_centroid=True)
        self.ablation = ablation

        # 只替换聚合层，其余结构不动
        self.hyp_aggregation_list = nn.ModuleList([
            build_aggregator(ablation,
                             hidden_size=self.feature_list[i + 1],
                             curvature_init=curvature_init)
            for i in range(self.stages)
        ])


# ============================================================================
# 命令行入口：单版本前向/反向冒烟测试
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFECNet 消融实验模型")
    parser.add_argument('--ablation', type=str, default='none',
                        choices=list(HFECNetAblation.VALID_ABLATIONS),
                        help='消融模式（默认 none，即完整双曲模型）')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--num_classes', type=int, default=101)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== 消融模式: {args.ablation} | device: {device} ===")

    model = HFECNetAblation(num_classes=args.num_classes,
                            ablation=args.ablation).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    # 前向 + 反向冒烟测试
    x = torch.randn(args.batch_size, 4, args.num_points, device=device)
    label = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    model.train()
    out = model(x)
    loss = F.cross_entropy(out, label)
    loss.backward()
    print(f"前向输出: {tuple(out.shape)}, loss={loss.item():.4f}")

    # 检查曲率参数梯度（fixed_curvature / no_hyperbolic 没有曲率参数）
    for name, p in model.named_parameters():
        if 'curvature' in name:
            print(f"曲率参数 {name}: value={p.item():.4f}, "
                  f"grad={'None' if p.grad is None else f'{p.grad.item():.2e}'}")
    print(f"✅ 消融模式 '{args.ablation}' 前向反向通过")
