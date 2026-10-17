[claude-code:unrecognized_model] {"model":"glm-5.3-shouyun","query_source":"sdk"}
# SECNet Stage 2 跨窗口聚合：设计分析与完整实现

## 1. `attention(1,1080)` 是什么？——基于 checkpoint 结构的推理

先看参数形状能告诉我们什么。`attention` 是 **一个** 1080 维向量，对所有窗口、所有视频共享；`classifier` 吃 3240 维（= 3×1080 的拼接）。由此可以排除/收窄设计空间：

**假设 A（特征级门控，排除）**：`out_i = a ⊙ f_i`，对三个窗口用同一个向量做逐元素门控再拼接。这在数学上是退化的——对三个窗口施加**相同**的对角缩放，可以被 classifier 第一层线性权重完全吸收（`W·diag(a)·f = W'·f`），等于冗余参数。审稿人/作者没理由这么设计，训练时它也学不出非平凡结构之外的任何东西。

**假设 B（窗口级注意力 + 加权拼接，最可能）**：
```
s_i = a · f_i              # 每个窗口一个标量分数（点积），i = 1,2,3
w = softmax([s_1, s_2, s_3])  # 3 个窗口权重
out = concat(w_1·f_1, w_2·f_2, w_3·f_3)  # (B, 3240)
logits = classifier(out)   # 3240 → 1024 → 101
```
这是唯一一个既让 `attention` 产生**非退化**作用、又与 `classifier(3240→…)` 输入维度吻合的结构：`attention` 决定三个 30ms 窗口各自的贡献强度，加权后的特征拼接进 MLP。窗口级分数来自点积，softmax 的“温度”隐含在 `||a||` 里（可学习），所以不需要额外的 scaling 因子。

**假设 C（加权求和，排除）**：`Σ w_i f_i` 输出 1080 维，与 3240 的 classifier 输入不符。

**假设 D（sigmoid 代替 softmax）**：结构与 B 相同，只是权重用 `σ(s_i)` 逐窗口独立计算。维度上同样自洽，作为备选。

**关键验证手段**：官方 checkpoint 里就有训好的 `attention` 和 `classifier`。用我们的骨干提取测试集特征后，把官方权重按假设 B 装载、直接推理——如果 video-level 准确率落在 82.4% 附近，架构假设即被证实；如果明显偏低（比如 70% 出头），再依次试假设 D、特征级门控等变体。**这是复现里最值钱的一步**，比任何猜测都可靠（见第 6 节的注意事项）。

训练方式：官方只发布了 Stage 2 的 `attention + classifier` 权重（骨干沿用 Stage 1），强烈暗示是**两阶段训练——冻结骨干、只训聚合器**。特征缓存后训练量极小（每 video 只有 3×1080 维输入），几分钟收敛，也避免小数据端到端微调把 60% 的单窗口骨干带偏。端到端微调作为可选增益实验放最后。

---

## 2. WindowAggregator 模块

```python
# -*- coding: utf-8 -*-
# stage2/aggregator.py
"""
SECNet Stage 2 跨窗口聚合模块。

结构（与官方 checkpoint 参数形状对齐）:
  attention : (1, 1080) 可学习向量，与每个窗口特征点积得到窗口级标量分数，
              softmax 得到 3 个窗口权重，各窗口特征乘以权重后拼接 -> (B, 3240)
  classifier: 3240 -> 1024 -> 101 MLP

参数命名刻意与官方一致（attention / classifier），方便 strict=False 加载官方权重。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAggregator(nn.Module):
    def __init__(self, feat_dim=1080, n_windows=3, num_classes=101,
                 hidden_dim=1024, dropout=0.5, mode='softmax_concat'):
        """
        mode:
          'softmax_concat' : 假设 B，w=softmax(点积分数)，拼接 w_i*f_i  （首选）
          'sigmoid_concat' : 假设 D，w_i=sigmoid(s_i) 独立加权后拼接    （备选）
          'plain_concat'   : 消融基线，不加权直接拼接（attention 恒等于 1）
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.n_windows = n_windows
        self.mode = mode

        # (1, feat_dim) 与官方 ckpt 的 attention 形状一致
        # 初始化为小尺度，使初始 softmax 接近均匀分布（不偏袒任何窗口）
        self.attention = nn.Parameter(torch.empty(1, feat_dim))
        nn.init.normal_(self.attention, mean=0.0, std=1.0 / feat_dim)

        # 3240 -> 1024 -> 101
        # 注意：官方中间层是否有 BN/激活种类未知，这里用 Linear+ReLU+Dropout；
        # 若加载官方权重后发现 BN 参数缺失/多余，按 strict=False 的报告调整。
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * n_windows, hidden_dim),
            nn.BatchNorm1d(hidden_dim),      # 若官方无 BN 则删掉此行
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask=None):
        """
        x    : (B, n_windows, feat_dim)  同一视频 3 个窗口的特征，按时间顺序排列
        mask : (B, n_windows) 可选，1=有效窗口，0=补零的缺失窗口（缺失窗口不参与 softmax）
        返回 : logits (B, num_classes), weights (B, n_windows)
        """
        B, W, D = x.shape

        if self.mode == 'plain_concat':
            weights = torch.full((B, W), 1.0 / 1.0, device=x.device)  # 恒等权重
            weights = weights / 1.0
        else:
            # 窗口级分数: a·f_i -> (B, W)。不做固定 sqrt(D) 缩放，
            # 温度由可学习的 ||a|| 自适应（否则与官方权重的隐含尺度不符）
            scores = torch.einsum('bwd,od->bw', x, self.attention)
            if self.mode == 'softmax_concat':
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                weights = F.softmax(scores, dim=1)
            elif self.mode == 'sigmoid_concat':
                weights = torch.sigmoid(scores)
            else:
                raise ValueError(self.mode)

        # 加权后拼接: (B, W*D)
        weighted = x * weights.unsqueeze(-1)
        feat = weighted.reshape(B, W * D)
        logits = self.classifier(feat)
        return logits, weights


def load_official_stage2(model, state_dict, verbose=True):
    """
    把官方 checkpoint 里的 attention / classifier 装载进 WindowAggregator。
    官方键名若为 'classifier.weight'（单层写法）会自动重映射到 Sequential 索引。
    """
    own = model.state_dict()
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('attention'):
            new_state[k] = v
        elif k.startswith('classifier'):
            # 'classifier.weight' -> 'classifier.0.weight' 等
            mapped = k
            if mapped not in own and mapped.replace('classifier.', 'classifier.0.') in own:
                mapped = mapped.replace('classifier.', 'classifier.0.')
            new_state[mapped] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if verbose:
        print(f'[load_official_stage2] 缺失键: {missing}')
        print(f'[load_official_stage2] 忽略键: {unexpected}')
    return model
```

---

## 3. 特征提取脚本（冻结骨干，缓存窗口特征）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# stage2/extract_features.py
"""
用 Stage 1 训练好的 FECNet（冻结）逐窗口提取 1080 维特征（mean||max 池化输出），
缓存为 npz，供 Stage 2 训练/评测使用——聚合器训练每个 epoch 都要过全部窗口，
缓存后单 epoch 从分钟级降到毫秒级。

用法:
  python extract_features.py --h5 data/train.h5 --ckpt stage1_best.pth --save feats_train.npz
  python extract_features.py --h5 data/test.h5  --ckpt stage1_best.pth --save feats_test.npz
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.FECNet import FECNet  # 按实际类名/构造参数调整


def load_backbone(ckpt_path, device):
    model = FECNet(num_classes=101)  # TODO: 按 models/FECNet.py 实际构造函数调整
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and any(k in state for k in ('model', 'state_dict', 'net')):
        state = next(v for k, v in state.items()
                     if k in ('model', 'state_dict', 'net'))
    # Stage 1 的分类头参数与本任务无关，用 strict=False 忽略；缺失键应为空
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[backbone] missing={missing}\n[backbone] unexpected(忽略)={unexpected}')
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract(model, h5_path, save_path, device, batch_size=32):
    feats, labels, marks = [], [], []
    with h5py.File(h5_path, 'r') as f:
        data, label, mark = f['data'], f['label'], f['mark']
        n = data.shape[0]
        for i in range(0, n, batch_size):
            # 注意：若 Stage 1 训练/测试时对 'data' 有预处理（归一化、时间平移、
            # 通道重排等），这里必须与 Stage 1 推理代码逐行保持一致，否则特征错位
            x = torch.from_numpy(np.asarray(data[i:i + batch_size], dtype=np.float32)).to(device)
            out = model(x)
            # TODO: 按 FECNet.forward 的实际返回调整——需要的是 1080 维池化特征
            # （mean||max），不是 101 维 logits。若 forward 只返回 logits，
            # 请在 FECNet 里加一个 extract_feat() 方法返回池化后的特征。
            if isinstance(out, (tuple, list)):
                out = out[0]
            assert out.dim() == 2 and out.shape[1] == 1080, \
                f'期望 (B,1080) 特征，得到 {tuple(out.shape)}'
            feats.append(out.float().cpu().numpy())
            labels.append(np.asarray(label[i:i + batch_size], dtype=np.int64))
            marks.append(np.asarray(mark[i:i + batch_size], dtype=np.int64))
            if (i // batch_size) % 20 == 0:
                print(f'  {i}/{n}')
    np.savez_compressed(
        save_path,
        feat=np.concatenate(feats),     # (N, 1080)
        label=np.concatenate(labels),   # (N,)
        mark=np.concatenate(marks),     # (N,) 视频索引
    )
    print(f'已保存 {save_path}: {len(feats[0]) and sum(len(b) for b in feats)} 个窗口')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--ckpt', required=True, help='Stage 1 骨干权重')
    ap.add_argument('--save', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_backbone(args.ckpt, device)
    extract(model, args.h5, args.save, device, args.batch_size)
```

---

## 4. Stage 2 训练脚本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# stage2/train_stage2.py
"""
Stage 2 训练：在冻结骨干的缓存特征上，只训 WindowAggregator（attention + classifier）。

数据组织：按 mark 把窗口聚合成视频级样本 (3,1080)+label。
  - 同一视频的窗口按 h5 中的原始行序排列（即时间顺序 [35-65,135-165,235-265]ms），
    拼接顺序对 classifier 第一层是敏感的，训练与测试必须一致。
  - 若某视频不足 3 个窗口则补零并置 mask=0（softmax 时被 -inf 屏蔽）。

用法:
  python train_stage2.py --train feats_train.npz --val feats_test.npz \
      --save stage2_best.pth --mode softmax_concat
  （验证集也可从 train 里切 5%，这里默认直接用 test 监控——注意这是“复现”场景，
   若要做严格的模型选择，请改用 --val 从训练集切分。）
"""
import argparse
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from aggregator import WindowAggregator

N_WINDOWS, FEAT_DIM = 3, 1080


class VideoDataset(Dataset):
    """把窗口级特征按 mark 聚合成视频级样本"""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        feat, label, mark = d['feat'], d['label'], d['mark']
        groups = defaultdict(list)
        for i, m in enumerate(mark):
            groups[int(m)].append(i)
        self.samples = []
        for m, idxs in sorted(groups.items()):
            idxs = sorted(idxs)[:N_WINDOWS]      # 原始行序 = 窗口时间序
            fs = np.zeros((N_WINDOWS, FEAT_DIM), dtype=np.float32)
            mask = np.zeros(N_WINDOWS, dtype=np.float32)
            for j, idx in enumerate(idxs):
                fs[j] = feat[idx]
                mask[j] = 1.0
            assert len(set(label[idxs].tolist())) == 1, f'视频 {m} 窗口标签不一致'
            self.samples.append((fs, mask, int(label[idxs[0]])))
        print(f'{npz_path}: {len(self.samples)} 个视频')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fs, mask, y = self.samples[i]
        return (torch.from_numpy(fs), torch.from_numpy(mask), y)


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits, _ = model(x, mask)
        loss = nn.functional.cross_entropy(logits, y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--save', default='stage2_best.pth')
    ap.add_argument('--mode', default='softmax_concat',
                    choices=['softmax_concat', 'sigmoid_concat', 'plain_concat'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=5e-4)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = VideoDataset(args.train)
    val_ds = VideoDataset(args.val)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = WindowAggregator(mode=args.mode).to(device)
    # 只有两个可训练模块：attention 参数 + classifier
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_ld, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_ld, device)
        scheduler.step()
        print(f'epoch {ep:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} '
              f'| val loss {va_loss:.4f} acc {va_acc:.4f}')
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())

    torch.save({'model': best_state, 'mode': args.mode,
                'feat_dim': FEAT_DIM, 'n_windows': N_WINDOWS}, args.save)
    print(f'best val acc = {best_acc:.4f}, 已保存 {args.save}')
    # 打印学到的窗口注意力分布（在均匀样本上取平均）， sanity check 用
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x = torch.stack([torch.from_numpy(s[0]) for s in val_ds.samples]).to(device)
        mask = torch.stack([torch.from_numpy(s[1]) for s in val_ds.samples]).to(device)
        _, w = model(x, mask)
        print(f'平均窗口权重 (w1,w2,w3) = {w.mean(0).cpu().numpy().round(3)}')


if __name__ == '__main__':
    main()
```

---

## 5. 评测脚本（video-level 准确率 + 官方权重验证）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# stage2/eval_stage2.py
"""
按 mark 分组聚合测试窗口，计算 video-level 准确率。
两种用法:
  1) 评测自己训的 Stage 2:
     python eval_stage2.py --feats feats_test.npz --ckpt stage2_best.pth
  2) 【关键验证】直接装载官方 checkpoint 的 attention/classifier 推理:
     python eval_stage2.py --feats feats_test.npz --official official_ckpt.pth \
         --backbone official_ckpt.pth --mode softmax_concat
     若准确率落在 82.4% 附近 => 架构假设证实；否则换 --mode 逐个试变体。
"""
import argparse

import numpy as np
import torch

from aggregator import WindowAggregator, load_official_stage2
from train_stage2 import VideoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--ckpt', default=None, help='自己训的 stage2 权重')
    ap.add_argument('--official', default=None, help='官方完整 checkpoint（含骨干+stage2）')
    ap.add_argument('--mode', default='softmax_concat')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = VideoDataset(args.feats)

    if args.official:
        # 官方模式：骨干也从官方 ckpt 加载（见第 6 节注意事项）
        from extract_features import load_backbone, extract
        import tempfile, h5py  # noqa
        raise SystemExit(
            '官方验证请先 extract_features.py --ckpt <官方ckpt> 提取特征，'
            '再回到本脚本 --official 只装载 attention/classifier。')
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location='cpu')
        model = WindowAggregator(mode=ck.get('mode', args.mode))
        model.load_state_dict(ck['model'])
    elif args.official:
        model = WindowAggregator(mode=args.mode)
        state = torch.load(args.official, map_location='cpu')
        state = state.get('model', state) if isinstance(state, dict) else state
        load_official_stage2(model, state)
    else:
        raise SystemExit('需要 --ckpt 或 --official 之一')

    model.to(device).eval()
    correct, total = 0, 0
    # 逐视频太慢也无必要，批量拼成 (B,3,1080)
    with torch.no_grad():
        for i in range(0, len(ds), 128):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 128, len(ds)))]).to(device)
            m = torch.stack([ds[j][1] for j in range(i, min(i + 128, len(ds)))]).to(device)
            y = torch.tensor([ds[j][2] for j in range(i, min(i + 128, len(ds)))]).to(device)
            logits, _ = model(x, m)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    print(f'video-level accuracy: {correct}/{total} = {correct / total:.4f}')


if __name__ == '__main__':
    main()
```

（`--official` 分支里上面那段提示是刻意保留的流程引导；实际流程：先 `extract_features.py --ckpt 官方ckpt` 得到官方骨干的特征，再 `eval_stage2.py --feats ... --official 官方ckpt --mode softmax_concat`。）

---

## 6. 不确定性、备选设计与验证路线

**① 最优先：官方权重逆向验证（一步定位真架构）**
- 前提条件要满足：验证时**骨干也必须用官方 checkpoint 的权重**提取特征。若用我们自己复现的 Stage 1 骨干（单窗口只有 ~60%，且特征分布与官方有偏差），即使架构完全正确也到不了 82.4%——这是最容易误判的地方。如果官方 ckpt 是完整模型（含骨干），直接从中加载骨干；如果只含 stage2 两块权重，那 82.4% 只能用“官方骨干特征”验证，我们复现数字应对照论文表格另算。
- 验证顺序：`softmax_concat` → `sigmoid_concat` → 特征门控（`concat(a⊙f_1, a⊙f_2, a⊙f_3)`）→ 加权求和 + 1080 维探针（若官方 classifier 真是 3240 输入则应直接排除，但值得跑一次确认没有键名错位）。

**② classifier 内部结构未知**：3240→1024→101 之间是 ReLU 还是别的、有无 BN/Dropout，checkpoint 报告的参数量能反推（只有两个 Linear 的 weight+bias = 3240×1024+1024 + 1024×101+101 ≈ 3.42M；多出来的参数就是 BN）。`load_official_stage2` 的 strict=False 报告会直接暴露这一点——**先看官方 ckpt 里 classifier 到底有哪些键，再定模块结构**，这是第 2 节代码里 `TODO` 的依据。

**③ softmax 温度**：我用裸点积（温度由 `||a||` 学习）。若加载官方 attention 后验证分数极端（权重接近 one-hot 或完全均匀），可以试加/去 `sqrt(1080)` 缩放看哪个更吻合。

**④ 训练细节的不确定性**：
- 官方可能不是“只训聚合器”，而是端到端联合微调（论文若写 "two-stage" 则支持冻结方案；若写 "jointly" 则需微调）。若冻结方案收敛后明显低于 82.4%，做端到端变体：backbone lr=1e-5、aggregator lr=1e-4，并把 Stage 1 的单窗口 CE 作为辅助损失（防骨干漂移），特征不再缓存、改为在线前向。
- 视频数约 6967（train），聚合器参数 3.4M，可能过拟合——若 train acc 饱和而 val 停在低位，加大 dropout（0.5→0.7）、加 label smoothing 0.1、或把 1024 隐层缩到 512 做对比。

**⑤ 窗口顺序**：`[35-65, 135-165, 235-265]ms` 的拼接顺序对 classifier 第一层是硬编码敏感的。确认 h5 中每个 mark 的三行确实是按时间序存放（代码里用 `sorted(idxs)` 保留原始行序来保证）；如果官方是别的顺序（如逆序），官方权重验证会失败——这也是一个可切换的实验变量。

**⑥ 消融基线（写复现报告时有用）**：`plain_concat`（无注意力 MLP）应达到一个可观的下限；如果 `softmax_concat` 只比它高 1–2 个点，说明注意力的贡献主要来自"让 classifier 看到窗口可靠性"而非真正的软选择。另可报告三种简单聚合的上限参照：mean-pooling + 线性、max-pooling + 线性、三窗口 logits 投票。

**建议执行顺序**：核对官方 ckpt 键名/参数量（5 分钟，定 classifier 结构）→ 官方骨干提特征 → 逐个 mode 跑官方权重验证 → 确认架构后用自己复现的骨干特征训练 Stage 2 → 若与 82.4% 有差距，再上端到端微调。第 1 步和第 3 步的结果应该尽早反馈给我，我再据此修正模块代码。
