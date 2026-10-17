# -*- coding: utf-8 -*-
"""SECNet Stage 2 跨窗口聚合模块（根据官方 checkpoint 结构逆向复现）

官方 checkpoint（best_checkpoint.pth，best_test_acc=82.377）顶层 state_dict 中
除 model.*（Stage 1 backbone，93 个参数张量）外还有两组 Stage 2 权重：

  attention.linear.weight [1, 1080] + bias [1]     窗口级注意力
  classifier.1.weight [1024, 3240]                 Linear(3240 -> 1024)
  classifier.2.*  BN1d(1024)
  classifier.5.weight [1024, 1024]                 Linear(1024 -> 1024)
  classifier.6.*  BN1d(1024)
  classifier.9.weight [101, 1024]                  Linear(1024 -> 101)

关键推理依据：
1. attention 是 Linear(1080, 1)：1080 = backbone mean||max 池化特征维度，
   输出 1 维标量 → 对每个窗口打一个分，softmax 后得到 3 个窗口权重。
   这是【窗口级注意力】，不可能是特征级（特征级需要 1080 维输出）。
   与 Stage 1 代码里 Attention 类（models/FECNet.py:96）完全同构——
   Stage 1 的 aggregation_list 也是用 Linear(C,1)+softmax 做组内加权池化。
   注意：Stage 1 里的 model.attention(540) 在 forward 中未被使用（死代码），
   但 Stage 2 顶层的 attention(1080) 是真正参与前向的。
2. classifier 输入 3240 = 3 窗口 × 1080 → 注意力不是把 3 窗口加权求和成
   1080 维（那样 classifier 输入应为 1080），而是对每个窗口的特征做
   注意力加权后【拼接】保留全部 3 个窗口的信息。
   具体加权方式（softmax 加权 / sigmoid 门控 / 不加权）官方未公开，
   由 stage2_probe_official.py 用官方权重在测试集上逐一验证，
   能复现 82.377% 的那个即为官方设计。
3. classifier Sequential 下标 1/2/5/6/9 有参数、0/3/4/7/8 无参数 →
   结构为 [Dropout, Linear, BN, GELU, Dropout, Linear, BN, GELU, Dropout, Linear]，
   与 Stage 1 classifier [Dropout, Linear, BN, GELU, Dropout, Linear] 同款
   模式加深一层（Stage 1 下标 1/2/5 有参数，位置完全对应）。
4. optimizer 状态佐证训练配方：SGD(momentum=0.9, 初始 lr=0.1, wd=1e-4)，
   105 个参数中【仅 12 个】有动量缓冲，且 shape 与 attention(2 个) +
   classifier 的 Linear weight/bias(6 个) + BN affine(4 个) 一一对应，
   backbone 93 个参数无动量缓冲 → 官方 Stage 2【冻结 backbone、只训聚合器】。
"""
import os
import sys

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


class Attention(nn.Module):
    """窗口注意力：Linear(D,1) 对每个窗口打分，softmax 得窗口权重。

    与 models/FECNet.py 的 Attention 类逐行一致（官方 Stage 2 复用了同一个类，
    checkpoint 键名 attention.linear.* 完全匹配）。
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, W, D]（W=窗口数）→ 权重 [B, W]
        attn_weights = self.linear(x).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs


def build_stage2_classifier(feat_dim=1080, num_windows=3, hidden_dim=1024,
                            num_classes=101, dropout=(0.1, 0.5, 0.5)):
    """Stage 2 分类头：3240 -> 1024 -> 1024 -> 101

    Sequential 下标必须与官方 checkpoint 对齐：
      0 Dropout / 1 Linear / 2 BN1d / 3 GELU / 4 Dropout /
      5 Linear / 6 BN1d / 7 GELU / 8 Dropout / 9 Linear
    Dropout 率官方未存（无参数），按 Stage 1 的 (0.1, 0.5) 模式外推为
    (0.1, 0.5, 0.5)，仅影响训练正则，不影响加载与推理。
    """
    in_dim = feat_dim * num_windows
    return nn.Sequential(
        nn.Dropout(dropout[0]),
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout[1]),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout[2]),
        nn.Linear(hidden_dim, num_classes),
    )


class WindowAggregator(nn.Module):
    """Stage 2 跨窗口聚合器：3 窗口特征 -> 注意力加权 -> 拼接 -> MLP -> 视频 logits

    forward_mode 候选设计（官方未公开，用官方权重探测确定）：
      attn_concat : x = concat([a1*f1, a2*f2, a3*f3])，a = softmax(Linear(f))
                    —— 主假设：注意力加权后拼接，保留全部窗口信息
      raw_concat  : x = concat([f1, f2, f3])，注意力不参与（对照组，
                    检验 attention 是否真的被用到）
      wsum_tile   : x = tile(Σ ai*fi)，加权求和成 1080 后复制 3 份凑 3240
      gate_concat : x = concat([g1*f1, g2*f2, g3*f3])，g = sigmoid(Linear(f))
                    无归一化的门控加权
      wsum_meanmax: x = concat([Σ ai*fi, max_i fi, mean_i fi])，加权求和
                    与跨窗口 max/mean 池化拼接
    """

    def __init__(self, feat_dim=1080, num_windows=3, hidden_dim=1024,
                 num_classes=101, forward_mode='attn_concat',
                 dropout=(0.1, 0.5, 0.5)):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_windows = num_windows
        self.forward_mode = forward_mode
        self.attention = Attention(feat_dim)
        self.classifier = build_stage2_classifier(
            feat_dim, num_windows, hidden_dim, num_classes, dropout)

    def forward(self, feats):
        """feats: [B, W, D] 按时间顺序排列的 W 个窗口特征（D=1080）"""
        B, W, D = feats.shape
        assert W == self.num_windows, f'期望 {self.num_windows} 窗口, 得到 {W}'
        mode = self.forward_mode

        if mode == 'raw_concat':
            # 注意力不参与，直接拼接（对照组）
            x = feats.reshape(B, W * D)
        elif mode == 'attn_concat':
            # 主假设：softmax 窗口权重逐窗口缩放后拼接
            probs = self.attention(feats)                    # [B, W]
            x = (probs.unsqueeze(-1) * feats).reshape(B, W * D)
        elif mode == 'gate_concat':
            # sigmoid 门控（无 softmax 归一化）加权后拼接
            gates = torch.sigmoid(self.attention.linear(feats).squeeze(-1))  # [B, W]
            x = (gates.unsqueeze(-1) * feats).reshape(B, W * D)
        elif mode == 'wsum_tile':
            # 加权求和成 1080 维，复制 3 份凑 3240 维输入
            probs = self.attention(feats)                    # [B, W]
            s = torch.bmm(probs.unsqueeze(1), feats).squeeze(1)  # [B, D]
            x = s.repeat(1, W)
        elif mode == 'wsum_meanmax':
            # 注意力加权和 || 跨窗口 max 池化 || 跨窗口 mean 池化
            probs = self.attention(feats)                    # [B, W]
            s = torch.bmm(probs.unsqueeze(1), feats).squeeze(1)  # [B, D]
            x = torch.cat([s, feats.max(dim=1).values, feats.mean(dim=1)], dim=1)
        else:
            raise ValueError(f'未知 forward_mode: {mode}')

        return self.classifier(x)


def load_official_stage2_state(ckpt_path):
    """从官方 checkpoint 提取 Stage 2 权重（顶层 attention.* / classifier.*）"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    net = ckpt.get('net', ckpt)
    sd = {k: v for k, v in net.items() if not k.startswith('model.')}
    return sd


def load_backbone(ckpt_path, num_classes=101):
    """加载 Stage 1 backbone（FECNet），兼容官方（model.* 前缀）与自训 ckpt。

    官方 checkpoint 的 net 键形如 model.embed_dim.*，即官方训练脚本把
    FECNet 包在 self.model 下（外层另有 attention/classifier 两个 Stage 2 模块）。
    自训 stage1 ckpt（train_frequency_ncaltech.py 产出）键无前缀，直接加载。

    Codex 修复（2026-10-16）：原版硬编码 FECNet(num_classes) 默认构造
    （全部组件 flag 关闭），但自训 ckpt 可能带 use_hyperbolic_selector 等
    组件 flag——默认构造的 forward 走标准 FPS 而非双曲网格，提取的特征
    （单窗 acc 64.67% vs 正确的 71.00%）语义错位。现从 ckpt 的 args 恢复
    全部组件 flag（缺键兜底 False，兼容官方旧格式无 args）。
    """
    sys.path.append(os.path.join(ROOT_DIR, 'models'))
    from FECNet import FECNet
    ckpt = torch.load(ckpt_path, map_location='cpu')
    net = ckpt.get('net', ckpt)
    if any(k.startswith('model.') for k in net):
        sd = {k[len('model.'):]: v for k, v in net.items() if k.startswith('model.')}
    else:
        sd = net
    # 从 ckpt.args 恢复组件 flag（自训 ckpt 有；官方旧格式无 → 全部 False）
    a = ckpt.get('args', {}) or {}
    model = FECNet(
        num_classes=num_classes,
        use_group_norm=bool(a.get('group_norm', False)),
        use_time_enc=bool(a.get('time_enc', False)),
        use_kda=bool(a.get('kda', False)),
        use_rp1_selector=bool(a.get('rp1_selector', False)),
        use_hyperbolic_selector=bool(a.get('hyperbolic_selector', False)),
        use_spike_time_enc=bool(a.get('spike_time_enc', False)),
        use_hyperbolic_classifier=bool(a.get('hyperbolic_classifier', False)),
        use_hyperbolic_fusion=bool(a.get('hyperbolic_fusion', False)),
        use_hyperbolic_embedding=bool(a.get('hyperbolic_embedding', False)),
        use_hyperbolic_attention=bool(a.get('hyperbolic_attention', False)),
        use_event_mamba_ssm=bool(a.get('event_mamba_ssm', False)),
        use_hyperbolic_mlp=bool(a.get('hyperbolic_mlp', False)),
        use_scale_norm_fps=bool(a.get('scale_norm_fps', False)),
        use_hsel_scaled_coords=bool(a.get('hsel_scaled_coords', False)),
    )
    # strict=True：官方/自训 ckpt 都应逐键对上
    model.load_state_dict(sd, strict=True)
    return model


if __name__ == '__main__':
    # 自检：结构对齐官方 checkpoint 键名/形状
    agg = WindowAggregator()
    sd = agg.state_dict()
    ref = load_official_stage2_state(os.path.join(
        ROOT_DIR, 'archive/old_checkpoints/checkpoint_official/best_checkpoint.pth'))
    ref_train = {k: v for k, v in ref.items()
                 if 'running' not in k and 'num_batches' not in k}
    mine = {k: v for k, v in sd.items()
            if 'running' not in k and 'num_batches' not in k}
    assert set(mine.keys()) == set(ref_train.keys()), (
        f'键不一致\n mine-only: {set(mine) - set(ref_train)}\n'
        f' ref-only: {set(ref_train) - set(mine)}')
    for k in ref_train:
        assert tuple(ref_train[k].shape) == tuple(mine[k].shape), f'{k} shape 不匹配'
    # 官方权重应能 strict 加载
    agg.load_state_dict(ref, strict=True)
    out = agg(torch.randn(4, 3, 1080))
    print(f'自检通过: {len(ref_train)} 个可训练键名/形状与官方一致, '
          f'官方权重 strict 加载成功, 前向输出 {tuple(out.shape)}')
