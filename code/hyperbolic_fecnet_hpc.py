"""
双曲增强版 FECNet (HFECNet)

在 SECNet/FECNet 的每个 stage 中，将欧氏注意力组内聚合替换为双曲聚合，
利用庞加莱双曲空间建模事件点组内的部件-整体层级拓扑。

核心模块：
  - HyperbolicGroupAggregator: 双曲组内聚合（双曲距离注意力 + 切空间加权聚合 + 双曲投影）
  - HFECNet: 双曲增强版 FECNet，替换原模型的 aggregation_list

设计参考：
  - EHGCN: 双曲图卷积事件分类（双曲空间建模层级拓扑）
  - SECNet/FECNet: 傅里叶频域 Point-based 事件分类
  - Poincaré Embeddings: 双曲空间嵌入基础
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

from models.FECNet import FECNet, Linear1Layer, Linear2Layer, fftlayer, Attention
from models.modules import LocalGrouper


# ============================================================================
# 双曲几何基础运算（精简版，从 ECAF-GCN hyperbolic.py 复用）
# ============================================================================

def clip_to_disk(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """把点投影回 Poincaré 球内部。"""
    max_norm = (1.0 / torch.sqrt(c)) - eps
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    cond = norm > max_norm
    scale = torch.where(cond, max_norm / norm, torch.ones_like(norm))
    return x * scale


def poincare_log_origin(y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对数映射（原点锚定）：双曲空间 → 切空间。"""
    norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(eps)
    arg = (torch.sqrt(c) * norm).clamp(max=1.0 - eps)
    coef = (2.0 / torch.sqrt(c)) * torch.arctanh(arg)
    return torch.where(norm <= eps, torch.zeros_like(y), coef * (y / norm))


def poincare_exp_origin(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """指数映射（原点锚定）：切空间 → 双曲空间。"""
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    coef = torch.tanh(torch.sqrt(c) * norm / 2.0) / (torch.sqrt(c) * norm)
    out = coef * v
    return torch.where(norm <= eps, torch.zeros_like(out), clip_to_disk(out, c))


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """双曲距离。x,y: [..., d]，返回 [...]。"""
    x_norm_sq = (x * x).sum(-1).clamp(max=(1.0 / c) - eps)
    y_norm_sq = (y * y).sum(-1).clamp(max=(1.0 / c) - eps)
    diff_sq = ((x - y) ** 2).sum(-1)
    z = 1 + 2 * c * diff_sq / ((1 - c * x_norm_sq) * (1 - c * y_norm_sq)).clamp_min(eps)
    z = z.clamp_min(1.0 + eps)
    arccosh_z = torch.log(z + torch.sqrt((z - 1.0) * (z + 1.0)))
    return (1.0 / torch.sqrt(c)) * arccosh_z


# ============================================================================
# 双曲组内聚合模块
# ============================================================================

class HyperbolicGroupAggregator(nn.Module):
    """双曲组内聚合：用双曲距离计算注意力权重，在切空间做加权聚合后投影回双曲空间。

    输入: [B*G, K, C]（K 个邻居点的特征）
    输出: [B*G, C]（聚合后的组代表特征）

    流程:
      1. 线性投影到合适维度（可选，保持维度不变）
      2. 指数映射到双曲空间
      3. 以组内第一个点（或中心点）为锚，计算双曲距离
      4. 双曲距离 → 注意力权重（softmax(-distance/temperature)）
      5. 对数映射回切空间，注意力加权聚合
      6. 指数映射回双曲空间，再对数映射回欧氏空间（保证输出在欧氏空间）
    """

    def __init__(self, hidden_size: int, curvature_init: float = 0.4,
                 temperature: float = 1.0, use_centroid: bool = True):
        """
        Args:
            hidden_size: 特征维度 C
            curvature_init: 双曲曲率初始值（论文 c ∈ {0.2,0.4,0.6,0.8}）
            temperature: 注意力温度系数
            use_centroid: True=用组内质心作为锚点，False=用第一个点作为锚点
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.curvature = nn.Parameter(torch.tensor(curvature_init))
        self.temperature = temperature
        self.use_centroid = use_centroid
        # Möbius 线性变换（参考 ECAF-GCN Hyperbolic Feature Transformation：
        # 特征在切空间做线性变换后映射回双曲，等价于 Möbius 矩阵乘法 W ⊗_c x）
        self.mobius_linear = nn.Linear(hidden_size, hidden_size)
        # Möbius 偏置（加到切空间变换结果上，等价于 ⊕_c b）
        self.mobius_bias = nn.Parameter(torch.zeros(hidden_size))
        # 输出投影（双曲聚合后可能需要微调）
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _curvature(self) -> torch.Tensor:
        """曲率需为正，用 abs+eps 保证（curvature_init=0.4 时实际曲率≈0.4）。"""
        return torch.abs(self.curvature) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*G, K, C]，K 个邻居点的欧氏特征
        Returns:
            [B*G, C]，聚合后的欧氏特征
        """
        c = self._curvature()
        K = x.shape[1]

        # 1. 指数映射到双曲空间
        x_hyp = poincare_exp_origin(x, c)  # [B*G, K, C]

        # 2. 确定锚点
        if self.use_centroid:
            # 组内质心（欧氏空间平均后映射到双曲空间）
            centroid_euclidean = x.mean(dim=1, keepdim=True)  # [B*G, 1, C]
            anchor = poincare_exp_origin(centroid_euclidean, c)  # [B*G, 1, C]
        else:
            anchor = x_hyp[:, 0:1, :]  # [B*G, 1, C]

        # 3. 计算双曲距离（锚点与每个点之间）
        anchor_expanded = anchor.expand_as(x_hyp)  # [B*G, K, C]
        distances = poincare_distance(anchor_expanded, x_hyp, c)  # [B*G, K]

        # 4. 双曲距离 → 注意力权重
        attn_weights = F.softmax(-distances / self.temperature, dim=-1)  # [B*G, K]

        # 5. Möbius 线性变换：特征真在双曲里变换
        #    切空间线性变换 W·log_o(x) + b，再指数映射回双曲（等价 W ⊗_c x ⊕_c b）
        x_tangent = poincare_log_origin(x_hyp, c)  # [B*G, K, C]
        x_transformed = self.mobius_linear(x_tangent) + self.mobius_bias  # [B*G, K, C]
        x_hyp_transformed = poincare_exp_origin(x_transformed, c)  # [B*G, K, C]

        # 6. 对数映射回切空间，注意力加权聚合（权重基于变换前的双曲距离）
        agg_tangent = torch.bmm(
            attn_weights.unsqueeze(1), poincare_log_origin(x_hyp_transformed, c)
        ).squeeze(1)  # [B*G, C]

        # 7. 指数映射回双曲空间，再对数映射回欧氏空间
        agg_hyp = poincare_exp_origin(agg_tangent, c)  # [B*G, C]
        agg_euclidean = poincare_log_origin(agg_hyp, c)  # [B*G, C]

        # 8. 输出投影 + 残差 + LayerNorm
        out = self.out_proj(agg_euclidean)  # [B*G, C]
        # 残差连接（与欧氏均值相加，保证梯度流畅）
        residual = x.mean(dim=1)  # [B*G, C]
        out = self.layer_norm(out + residual)

        return out


# ============================================================================
# 双曲增强版 FECNet
# ============================================================================

class HFECNet(nn.Module):
    """双曲增强版 FECNet。

    与原 FECNet 的唯一区别：将每个 stage 的欧氏注意力聚合（Attention + bmm）
    替换为双曲组内聚合（HyperbolicGroupAggregator）。

    其余结构（embed_dim, LocalGrouper, 空间FFT, 时间FFT, Linear2Layer, classifier）
    完全保持不变，确保可以加载预训练权重做微调。
    """

    def __init__(self, num_classes: int, curvature_init: float = 0.4,
                 use_centroid: bool = True):
        super().__init__()
        # 官方 N-Caltech101 配置（与 SECNet 论文 best_checkpoint 一致，公平对比）
        self.feature_list = [64, 132, 268, 540]
        self.group_number = [2048, 1024, 512, 256]
        self.neighbors = [24, 24, 24, 24]
        self.stages = 3

        self.embed_dim = Linear1Layer(4, self.feature_list[0], 1)
        self.local_grouper_list = nn.ModuleList()
        self.hyp_aggregation_list = nn.ModuleList()  # 双曲聚合（替换原 Attention）
        self.fft_sptaial_list = nn.ModuleList()
        self.fft_temporal_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.attention = Attention(self.feature_list[-1])

        for i in range(self.stages):
            local_grouper = LocalGrouper(self.feature_list[i], self.group_number[i], 24, True, "center")
            self.local_grouper_list.append(local_grouper)

            # 双曲组内聚合（替换原 Attention(self.feature_list[i+1]) + bmm）
            hyp_agg = HyperbolicGroupAggregator(
                hidden_size=self.feature_list[i + 1],
                curvature_init=curvature_init,
                use_centroid=use_centroid,
            )
            self.hyp_aggregation_list.append(hyp_agg)

            # 空间 FFT 权重
            fft_sptaial_weight = nn.Parameter(torch.cat(
                (torch.ones(self.feature_list[i + 1] // 2 + 1, self.neighbors[i], 1, dtype=torch.float32),
                 torch.zeros(self.feature_list[i + 1] // 2 + 1, self.neighbors[i], 1, dtype=torch.float32)),
                dim=-1))
            self.fft_sptaial_list.append(fftlayer(fft_sptaial_weight))

            # 时间 FFT 权重
            fft_temporal_weight = nn.Parameter(torch.cat(
                (torch.ones(self.group_number[i] // 2 + 1, self.feature_list[i + 1], 1, dtype=torch.float32),
                 torch.zeros(self.group_number[i] // 2 + 1, self.feature_list[i + 1], 1, dtype=torch.float32)),
                dim=-1))
            self.fft_temporal_list.append(fftlayer(fft_temporal_weight))

            conv = Linear2Layer(self.feature_list[i + 1], 1, 1)
            self.conv_list.append(conv)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_list[-1] * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, N]，4=(x,y,t,p)，N=点数（8192）
        Returns:
            [B, num_classes] 分类 logits
        """
        xyz = x.permute(0, 2, 1)  # [B, N, 4]
        batch_size, _, _ = x.size()
        x = self.embed_dim(x)  # [B, 32, N]
        x = x.permute(0, 2, 1)  # [B, N, 32]

        for i in range(self.stages):
            # LocalGrouper: FPS 采样 + KNN 分组
            xyz, x = self.local_grouper_list[i](xyz, x)
            # x: [B, G, K, C]，G=组数，K=邻居数，C=feature_list[i+1]

            x = x.permute(0, 1, 3, 2)  # [B, G, C, K]
            b, n, d, s = x.size()
            x = x.reshape(-1, d, s)  # [B*G, C, K]

            # 空间 FFT（在 K 维度）
            x = self.fft_sptaial_list[i](x)
            x = F.gelu(x)
            x = x.permute(0, 2, 1)  # [B*G, K, C]

            # === 关键区别：双曲组内聚合（替换原 Attention + bmm）===
            x = self.hyp_aggregation_list[i](x)  # [B*G, C]

            x = x.reshape(b, n, -1)  # [B, G, C]

            # 时间 FFT（在 G 维度）
            x = self.fft_temporal_list[i](x)
            x = x.permute(0, 2, 1)  # [B, C, G]
            x = F.gelu(x)
            x = self.conv_list[i](x)  # Linear2Layer 残差
            x = x.permute(0, 2, 1)  # [B, G, C]

        # 全局池化（均值 + 最大）
        x_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1)
        x_max = x_max.reshape(batch_size, -1)
        x_mean = x.mean(dim=1)
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.classifier(x)
        return x

    def load_pretrained_fecnet(self, state_dict: dict):
        """从原 FECNet 预训练权重加载（跳过双曲聚合层，其余层一一对应）。

        Args:
            state_dict: 原 FECNet 的 state_dict
        """
        own_state = self.state_dict()
        loaded = 0
        skipped = []
        for name, param in state_dict.items():
            if name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
                else:
                    skipped.append(f"{name} (shape mismatch: {param.shape} vs {own_state[name].shape})")
            else:
                skipped.append(f"{name} (not in HFECNet)")
        print(f"Loaded {loaded}/{len(state_dict)} parameters from FECNet")
        if skipped:
            print(f"Skipped {len(skipped)} layers:")
            for s in skipped[:10]:
                print(f"  - {s}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=== HFECNet 测试 ===")
    model = HFECNet(num_classes=101, curvature_init=0.4)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # 统计双曲参数占比
    hyp_params = 0
    for name, p in model.named_parameters():
        if "hyp_aggregation" in name or "curvature" in name:
            hyp_params += p.numel()
    print(f"Hyperbolic params: {hyp_params:,} ({100*hyp_params/total_params:.2f}%)")

    print("\n=== 前向传播测试 ===")
    batch_size = 2
    num_points = 8192
    x = torch.randn(batch_size, 4, num_points)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        try:
            output = model(x)
            print(f"Output shape: {output.shape}")
            print(f"Output sample (first 5): {output[0, :5]}")
            print("\n✅ HFECNet 前向传播成功！")
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
        loss = F.cross_entropy(output, label)
        loss.backward()
        print(f"Loss: {loss.item():.4f}")
        # 检查曲率梯度
        for name, p in model.named_parameters():
            if "curvature" in name:
                print(f"Curvature '{name}' value: {p.item():.4f}, grad: {p.grad.item() if p.grad is not None else 'None'}")
        print("✅ 反向传播成功！")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
