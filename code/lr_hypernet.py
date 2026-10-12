"""
LR-HyperNet: Local+Remote 邻域 + 远程邻域双曲聚合的事件分类网络

基于 SECNet/FECNet backbone，仅将每个 stage 的 LocalGrouper 替换为
LocalRemoteGrouper（局部 KNN + 远程双曲聚合），其余结构（embed_dim、
空间/时间 FFT、注意力聚合、门控融合、分类头）与 FECNet 完全一致：
  - 保证与 SECNet 基线的可控对比（唯一变量是分组器）
  - 分组器输出维度 2C+4 与原 LocalGrouper 相同，形状上兼容 FECNet 权重
"""
from __future__ import annotations

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

from models.FECNet import (FECNet, Linear1Layer, Linear2Layer, fftlayer,
                           Attention, make_norm1d)
from models.local_remote_grouper import LocalRemoteGrouper, HyperbolicPrototypeClassifier


class LRHyperNet(nn.Module):
    """Local+Remote 邻域双曲版 FECNet。

    与原 FECNet 的唯一区别：local_grouper_list 中的 LocalGrouper 替换为
    LocalRemoteGrouper（远程分支引入可学习曲率的双曲聚合，曲率初值 0.4）。
    """

    def __init__(self, num_classes: int, feature_list=None, group_number=None,
                 remote_g: int = 16, curvature_init: float = 0.4,
                 time_enc: bool = False, use_global_anchor: bool = False,
                 anchor_lambda: float = 0.5, kda: bool = False,
                 kda_bidirectional: bool = False,
                 hyperbolic_type: str = "poincare",
                 use_group_norm: bool = False,
                 hyperbolic_classifier: bool = False,
                 use_hyperbolic_selector: bool = False,
                 remote_dropout: float = 0.0,
                 use_rp1_selector: bool = False,
                 rp1_lambda: float = 1.0,
                 selector_mode: str = "topk",
                 div_lambda: float = 1.0,
                 remote_gate: bool = False,
                 remote_kda: bool = False,
                 density_channel: bool = False,
                 density_bins: int = 32):
        """
        Args:
            num_classes: 分类数（N-Caltech101 为 101）
            feature_list: 各 stage 特征维度（默认官方大配置 [64, 132, 268, 540]）
            group_number: 各 stage FPS 组数（默认 [2048, 1024, 512, 256]）
            remote_g: 每个中心点的远程邻居数 G
            curvature_init: 双曲聚合器曲率初值
            time_enc: 时间因果序编码 v2（EventMamba/Event2Vec 批判的直接回应）。
                对事件时间戳 t 做多频正弦位置编码（4 个频率的 sin/cos + 原 t），
                与 (x, y, p) 拼成 12 维输入嵌入，使双曲/远程聚合保留时间因果序。
                与 kda 互斥（二选一）。
            use_global_anchor: 全局吸收节点（PVAG 启发），远程选择混合全局锚点距离
            anchor_lambda: 全局锚点混合权重
            kda: KDA 因果时间编码 v3（Kimi Delta Attention 式 gated delta rule）。
                局部邻居按组内时间戳因果扫描，时间信息由递推状态隐式建模
                （替代显式 sin/cos 位置编码），输出维度不变。与 time_enc 互斥。
            kda_bidirectional: KDA 双向因果递推（v3.1，HydraMamba 启发）。
                仅 kda=True 时生效，前向+后向各跑一次后 Linear 融合。
            hyperbolic_type: "poincare"（庞加莱球，原版）、"lorentz"（洛伦兹模型，
                v3.1 新增，消除边界梯度尖峰）或 "euclidean"（欧氏等价对照，
                消融实验用：同 backbone/同分组/同参数量，仅把双曲远程聚合
                换成欧氏距离注意力，验证双曲增益）
            hyperbolic_classifier: 双曲原型分类头（路线A）。backbone 用欧氏远程聚合，
                分类头换成 Lorentz 双曲原型分类器。双曲增益在文献中最常出现在
                分类层而非聚合层，N-Caltech101 类别有 WordNet 层级结构。
            use_group_norm: 全网 BatchNorm1d → GroupNorm（b8 修复）。
                b8 下 BN 批统计量噪声极大（长尾类每 epoch 只见几个样本），
                GroupNorm 统计量与 batch size 无关。注意 LRHyperNet 内部的
                LocalRemoteGrouper/KDA/HyperbolicRemoteAggregator 均不含 BN
                （只用 LayerNorm），无需处理。
            use_hyperbolic_selector: 双曲距离选16个远程点（学长建议）。
                RemoteSelector 中先 exp map 到双曲空间，再用双曲距离
                代替欧氏距离选远程邻居。与 hyperbolic_type 配合使用
                （lorentz 推荐，数值更稳定）。
            use_rp1_selector: RP1 空间新颖性远程选择器（消融实验）。
                True 时远程选择得分 = 特征距离 + rp1_lambda * 空间新颖性
                （候选点与局部邻居的最小空间欧氏距离，RP1 论文：结合
                spatial novelty 的选择器优于纯特征距离 top-k，
                Oracle 68% vs 简单 KNN 61%）。False 时行为与原版完全一致。
            rp1_lambda: 空间新颖性权重（仅 use_rp1_selector=True 时有效）
            selector_mode: 远程点选择模式，"topk"（独立打分，默认）或 "greedy_div"（贪心最大-最小多样性，RP1启发）
            div_lambda: 多样性权重（仅 selector_mode="greedy_div" 时有效），越大远程点越分散
            remote_gate: SECNet式远程分支门控融合。中心点特征生成sigmoid门控，乘性调制远程特征（非加性），初始化bias=2.0起步等价直连
            remote_kda: EventMamba启发的远程KDA扫描。复用KDACausalEncoder对远程16点按时间排序做因果递推，捕获长程时序演变，残差加法
            density_channel: 密度输入通道（SpikePoint启发的rate coding密度信息）。
                对每个事件统计其所在 32x32 空间体素内的事件数并 log1p 压缩，
                作为第 5 个输入通道拼进 embed 输入。事件相机里"同坐标连续激发
                （边缘/运动）vs 孤立单发（噪声）"是截然不同的信号，密度通道
                让嵌入层直接感知局部激发率。默认 False 保证旧实验可复现。
            density_bins: 密度统计的空间体素边长（默认 32，即 32x32 体素）
        """
        super().__init__()
        if time_enc and kda:
            raise ValueError("time_enc 与 kda 互斥，只能二选一")
        # 官方 N-Caltech101 配置（与 SECNet 论文 best_checkpoint 一致，公平对比）
        self.feature_list = feature_list if feature_list is not None else [64, 132, 268, 540]
        self.group_number = group_number if group_number is not None else [2048, 1024, 512, 256]
        self.neighbors = [24, 24, 24, 24]
        self.stages = 3
        self.time_enc = time_enc
        self.kda = kda
        self.hyperbolic_type = hyperbolic_type
        self.density_channel = density_channel
        self.density_bins = density_bins

        # 时间因果序编码时输入维度 = x,y,p(3) + t(1) + 4 个频率 sin/cos(8) = 12
        # KDA/v1 路径输入维度 = 4（(t,x,y,p) 原样）
        embed_in_dim = 12 if time_enc else 4
        # 密度通道开启时各路径输入维度 +1
        if density_channel:
            embed_in_dim += 1
        self.embed_dim = Linear1Layer(embed_in_dim, self.feature_list[0], 1,
                                      use_group_norm=use_group_norm)
        self.local_grouper_list = nn.ModuleList()
        self.fft_temporal_weight_list = nn.ParameterList()
        self.fft_sptaial_weight_list = nn.ParameterList()
        self.fft_temporal_list = nn.ModuleList()
        self.fft_sptaial_list = nn.ModuleList()
        self.group_conv_list = nn.ModuleList()
        self.aggregation_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.bn_local_list = nn.ModuleList()
        self.bn_global_list = nn.ModuleList()
        self.attention = Attention(self.feature_list[-1])
        for i in range(self.stages):
            # === 关键区别：LocalRemoteGrouper 替换 LocalGrouper ===
            local_grouper = LocalRemoteGrouper(
                self.feature_list[i], self.group_number[i], 24, remote_g,
                True, "center", curvature_init=curvature_init,
                use_global_anchor=use_global_anchor, anchor_lambda=anchor_lambda,
                kda=kda, kda_bidirectional=kda_bidirectional,
                hyperbolic_type=hyperbolic_type,
                use_hyperbolic_selector=use_hyperbolic_selector,
                remote_dropout=remote_dropout,
                use_rp1_selector=use_rp1_selector,
                rp1_lambda=rp1_lambda,
                selector_mode=selector_mode,
                div_lambda=div_lambda,
                remote_gate=remote_gate,
                remote_kda=remote_kda)
            self.local_grouper_list.append(local_grouper)
            aggregation = Attention(self.feature_list[i + 1])
            self.aggregation_list.append(aggregation)
            fft_sptaial_weight = nn.Parameter(torch.cat(
                (torch.ones(self.feature_list[i + 1] // 2 + 1, self.neighbors[i], 1, dtype=torch.float32),
                 torch.zeros(self.feature_list[i + 1] // 2 + 1, self.neighbors[i], 1, dtype=torch.float32)),
                dim=-1))
            self.fft_sptaial_list.append(fftlayer(fft_sptaial_weight))
            fft_temporal_weight = nn.Parameter(torch.cat(
                (torch.ones(self.group_number[i] // 2 + 1, self.feature_list[i + 1], 1, dtype=torch.float32),
                 torch.zeros(self.group_number[i] // 2 + 1, self.feature_list[i + 1], 1, dtype=torch.float32)),
                dim=-1))
            self.fft_temporal_list.append(fftlayer(fft_temporal_weight))
            conv = Linear2Layer(self.feature_list[i + 1], 1, 1, use_group_norm=use_group_norm)
            self.conv_list.append(conv)
            group_conv = Linear2Layer(self.feature_list[i + 1], 1, 1, use_group_norm=use_group_norm)
            self.group_conv_list.append(group_conv)
            self.bn_local_list.append(make_norm1d(self.feature_list[i + 1], use_group_norm))
            self.bn_global_list.append(make_norm1d(self.feature_list[i + 1], use_group_norm))
        if hyperbolic_classifier:
            # 路线A：双曲原型分类头（欧氏backbone + 双曲分类层）
            self.classifier = HyperbolicPrototypeClassifier(
                input_dim=self.feature_list[-1] * 2,
                num_classes=num_classes,
                embed_dim=64,
                curvature_init=curvature_init
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.feature_list[-1] * 2, 256),
                make_norm1d(256, use_group_norm),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, N]，4=(t,x,y,p)（数据生成脚本 official_gen 的布局），N=点数（10240）
        Returns:
            [B, num_classes] 分类 logits
        """
        xyz = x.permute(0, 2, 1)  # [B, N, 4]（原样传给 grouper，保持 4 维语义）
        batch_size, _, _ = x.size()
        # 密度输入通道（SpikePoint启发的rate coding密度信息）：
        # 统计每个事件所在空间体素内的事件数并 log1p 压缩长尾。
        # 同坐标连续激发（边缘/运动）密度高，孤立单发（噪声）密度低，
        # 给嵌入层显式提供局部激发率信息。仅拼进 embed 输入，xyz 保持 4 维。
        density = None
        if self.density_channel:
            # x: [B, 4, N]，t,x,y 已归一化到 [0,1]
            bins = self.density_bins
            B_, _, N_ = x.shape
            xi = (x[:, 1] * bins).long().clamp(0, bins - 1)   # [B, N]
            yi = (x[:, 2] * bins).long().clamp(0, bins - 1)   # [B, N]
            vox = xi * bins + yi                              # [B, N] 展平体素索引
            count = torch.zeros(B_, bins * bins, device=x.device)
            count.scatter_add_(1, vox, torch.ones_like(vox, dtype=torch.float))
            density = count.gather(1, vox).log1p()            # [B, N]，log压缩长尾
        embed_in = x
        if self.time_enc:
            # 时间因果序编码：对 t（通道0）做多频正弦位置编码（Transformer 式），
            # 与 x,y,p 拼成 12 维 embed 输入；xyz 保持 [B,N,4] 供 grouper 使用
            t = xyz[..., 0:1]   # [B, N, 1] 时间戳，已归一化到 [0,1]
            xy = xyz[..., 1:3]  # [B, N, 2] x,y 坐标
            p = xyz[..., 3:4]   # [B, N, 1] 极性 ∈ {-1,+1}
            parts = [t]
            for i in range(4):
                freq = 2.0 ** i
                parts.append(torch.sin(2 * math.pi * freq * t))
                parts.append(torch.cos(2 * math.pi * freq * t))
            t_enc = torch.cat(parts, dim=-1)  # [B, N, 9] = t + 4sin + 4cos
            if density is not None:
                # 密度通道同步拼进 time_enc 路径：[B, N, 13] = xy,p + t_enc + density
                embed_in = torch.cat(
                    [xy, p, t_enc, density.unsqueeze(-1)], dim=-1).permute(0, 2, 1)
            else:
                embed_in = torch.cat([xy, p, t_enc], dim=-1).permute(0, 2, 1)  # [B, 12, N]
        elif density is not None:
            embed_in = torch.cat([x, density.unsqueeze(1)], dim=1)  # [B, 5, N]
        x = self.embed_dim(embed_in)
        x = x.permute(0, 2, 1)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x)
            x = x.permute(0, 1, 3, 2)
            b, n, d, s = x.size()
            x = x.reshape(-1, d, s)
            x = self.fft_sptaial_list[i](x)
            x = F.gelu(x)
            x = x.permute(0, 2, 1)
            att = self.aggregation_list[i](x)
            x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
            x = x.reshape(b, n, -1)
            x = self.fft_temporal_list[i](x)
            x = x.permute(0, 2, 1)
            x = F.gelu(x)
            # local branch
            x_local = self.conv_list[i](x)
            x_local = self.bn_local_list[i](x_local)
            # global branch: mean pool -> conv -> expand -> gate
            x_global = F.adaptive_avg_pool1d(x, 1)
            x_global = self.group_conv_list[i](x_global)
            x_global = self.bn_global_list[i](x_global)
            x_global = x_global.expand(-1, -1, n)
            # gated fusion
            x = x_local * torch.sigmoid(x_global)
            x = x.permute(0, 2, 1)
        x_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1)
        x_max = x_max.reshape(batch_size, -1)
        x = x.mean(dim=1)
        x = torch.cat([x, x_max], dim=1)
        x = self.classifier(x)
        return x

    def load_pretrained_fecnet(self, state_dict: dict) -> None:
        """从原 FECNet/SECNet 预训练权重加载（跳过远程双曲聚合新参数）。

        分组器输出维度与原 LocalGrouper 一致，fft/conv/分类头均可一一对应加载；
        仅远程双曲聚合器（hyper_agg.*）为新增参数，随机初始化。

        Args:
            state_dict: 原 FECNet 的 state_dict
        """
        own_state = self.state_dict()
        loaded = 0
        skipped = []
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped.append(name)
        print(f"Loaded {loaded}/{len(state_dict)} parameters from FECNet")
        if skipped:
            print(f"Skipped {len(skipped)} layers (new hyperbolic-remote params):")
            for s in skipped[:10]:
                print(f"  - {s}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    print("=== LR-HyperNet 测试 ===")
    for cfg_name, cfg in [
        ("v1 (baseline)", dict(curvature_init=0.4)),
        ("v2 (time_enc+anchor)", dict(curvature_init=0.4, time_enc=True, use_global_anchor=True)),
        ("v3 (kda+anchor)", dict(curvature_init=0.4, kda=True, use_global_anchor=True)),
        ("v3.1-euclidean (消融对照)", dict(time_enc=True, hyperbolic_type="euclidean")),
    ]:
        print(f"\n===== {cfg_name} =====")
        model = LRHyperNet(num_classes=101, **cfg)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        # 统计远程双曲分支参数占比
        remote_params = sum(p.numel() for n, p in model.named_parameters()
                            if "hyper_agg" in n or "remote" in n)
        print(f"Remote-hyperbolic params: {remote_params:,} ({100 * remote_params / total_params:.2f}%)")
        kda_params = sum(p.numel() for n, p in model.named_parameters() if "kda_enc" in n)
        if kda_params:
            print(f"KDA-encoder params: {kda_params:,} ({100 * kda_params / total_params:.2f}%)")

        print("--- 前向传播测试 ---")
        batch_size = 2
        num_points = 10240
        x = torch.randn(batch_size, 4, num_points)
        print(f"Input shape: {x.shape}")
        with torch.no_grad():
            output = model(x)
            print(f"Output shape: {output.shape}")
            assert output.shape == (2, 101), f"输出形状错误: {output.shape}"
            print(f"Output finite: {torch.isfinite(output).all().item()}")

        print("--- 反向传播测试 ---")
        model.train()
        x = torch.randn(batch_size, 4, num_points)
        label = torch.randint(0, 101, (batch_size,))
        output = model(x)
        loss = F.cross_entropy(output, label)
        loss.backward()
        print(f"Loss: {loss.item():.4f}")
        # 检查所有参数梯度无 NaN/Inf，并打印曲率/KDA 梯度
        bad_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f"非有限梯度的参数: {bad_grads if bad_grads else '无'}")
        for n, p in model.named_parameters():
            if "curvature" in n and p.grad is not None:
                print(f"Curvature '{n}': value={p.item():.4f}, grad={p.grad.item():.4f}")
            if "kda_enc" in n and "bias" in n and p.grad is not None:
                print(f"KDA '{n}': grad_finite={torch.isfinite(p.grad).all().item()}")
        assert not bad_grads, "存在 NaN/Inf 梯度"
        print(f"✅ {cfg_name} 前向/反向传播测试通过！")

    # 与 FECNet 对比
    fecnet_params = sum(p.numel() for p in FECNet(num_classes=101).parameters())
    model_v1 = LRHyperNet(num_classes=101)
    model_v3 = LRHyperNet(num_classes=101, kda=True, use_global_anchor=True)
    print(f"\nFECNet params: {fecnet_params:,}")
    print(f"LR-HyperNet v1 params: {sum(p.numel() for p in model_v1.parameters()):,} "
          f"(ratio: {sum(p.numel() for p in model_v1.parameters()) / fecnet_params:.2f}x)")
    print(f"LR-HyperNet v3 params: {sum(p.numel() for p in model_v3.parameters()):,} "
          f"(ratio: {sum(p.numel() for p in model_v3.parameters()) / fecnet_params:.2f}x)")
