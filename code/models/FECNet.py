import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LocalGrouper
from typing import Any, Callable, List, Optional, Union
from numbers import Number
import numpy as np


def _largest_group_count(channels: int, max_groups: int = 32) -> int:
    """取不超过 max_groups 且能整除 channels 的最大分组数。

    GroupNorm 要求 channels % num_groups == 0，而本网络通道数（132/268/540/66/134/270
    等）大多不是 32 的倍数，故从 min(32, C) 向下搜索第一个合法因子。
    """
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


def make_norm1d(channels: int, use_group_norm: bool = False) -> nn.Module:
    """归一化层工厂：默认 BatchNorm1d，--group_norm 时换 GroupNorm。

    b8 小 batch 下 BatchNorm 的批统计量（均值/方差）仅由 8 个样本估计，噪声极大，
    是 b8 变体严重欠拟合的嫌疑之一；GroupNorm 按 channel 分组归一化，统计量
    与 batch size 无关，小 batch 下行为稳定。
    """
    if use_group_norm:
        return nn.GroupNorm(_largest_group_count(channels), channels)
    return nn.BatchNorm1d(channels)


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True,
                 use_group_norm: bool = False):
        super(Linear1Layer, self).__init__()
        self.act = nn.GELU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            make_norm1d(out_channels, use_group_norm),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True,
                 use_group_norm: bool = False):
        super(Linear2Layer, self).__init__()
        self.act = nn.GELU()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels / 2),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            make_norm1d(int(in_channels / 2), use_group_norm),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=int(in_channels / 2), out_channels=in_channels,
                      kernel_size=kernel_size, bias=bias),
            make_norm1d(in_channels, use_group_norm)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class fftlayer(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.complex_weight = weight

    def forward(self, x):
        x = torch.fft.rfft(x, dim=(1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, dim=(1), norm='ortho')
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = self.linear(lstm_output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs


class FECNet(nn.Module):
    def __init__(self, num_classes, feature_list=None, group_number=None,
                 use_group_norm: bool = False):
        super().__init__()
        # 官方 N-Caltech101 配置（与 best_checkpoint 维度一致）
        self.feature_list = feature_list if feature_list is not None else [64, 132, 268, 540]
        self.group_number = group_number if group_number is not None else [2048, 1024, 512, 256]
        self.neighbors = [24, 24, 24, 24]
        self.stages = 3
        self.local_grouper_list = nn.ModuleList()
        self.fft_temporal_weight_list = nn.ParameterList()
        self.fft_sptaial_weight_list = nn.ParameterList()
        self.fft_temporal_list = nn.ModuleList()
        self.fft_sptaial_list = nn.ModuleList()
        self.embed_dim = Linear1Layer(4, self.feature_list[0], 1, use_group_norm=use_group_norm)
        self.group_conv_list = nn.ModuleList()
        self.aggregation_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.bn_local_list = nn.ModuleList()
        self.bn_global_list = nn.ModuleList()
        self.attention = Attention(self.feature_list[-1])
        for i in range(self.stages):
            local_grouper = LocalGrouper(self.feature_list[i], self.group_number[i], 24, True, "center")
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
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_list[-1] * 2, 256),
            make_norm1d(256, use_group_norm),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embed_dim(x)
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


def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """Count flops for the rfft/rfftn operator."""
    input_shape = inputs[0].type().sizes()
    print(input_shape)
    B, N, C = input_shape
    flops = N * C * np.ceil(np.log2(N)) + N * C * 2
    flops = flops * B
    return flops


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    model = FECNet(num_classes=10)
    model.eval()
    input = torch.randn(1, 4, 1024)
    fca1 = FlopCountAnalysis(model, input)
    handlers = {
        'aten::fft_rfft': rfft_flop_jit,
        'aten::fft_irfft': rfft_flop_jit,
    }
    fca1.set_op_handle(**handlers)
    flops1 = fca1.total()
    print(fca1.by_module())
    print(flop_count_table(fca1, max_depth=1))
    print("#### GFLOPs: {}".format(flops1 / 1e9))
