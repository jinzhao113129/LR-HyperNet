"""ECAF-GCN 模型（ICLR 2027 审稿论文复现）。

架构（论文 Fig.1 + 式 1-8）：
  事件时空图 G → 共享时空编码器（SplineConv）→ 双分支：
    ├─ 欧氏分支：多层 B-spline GCN（式 1），捕获局部结构 / 短程时空依赖
    └─ 双曲分支：指数映射（式 2）→ 多层 HyperbolicGCN（式 3-5），捕获全局层级 / 长程关系
  跨空间多头注意力（式 6-8）：双曲特征 log 到切空间作 Q，欧氏特征作 K/V，
      按全局结构语义选择性检索局部证据 → 融合
  → 空间网格池化（8×6）→ YOLOX 解耦检测头（回归 / 置信度 / 分类三分支）

实现说明：
  - 欧氏分支用 torch_geometric 的 SplineConv（与 AEGNN 同款，kernel_size=8, dim=3 边属性）。
  - 维度、层数、注意力头数等论文未公开的参数使用合理默认值，均标注【假设】。
  - 检测网格 8×6 与 AEGNN 一致（240×180 输入 → 30×30 cell）。
  - YOLOX head 采用 anchor-free 解耦结构（回归/置信度/分类三分支），
    完整 SimOTA 标签分配与 FPN 多尺度暂未实现【简化】。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from spline_conv import SplineConvPure as SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import scatter

from hyperbolic import HyperbolicGraphConv, poincare_exp_origin, poincare_log_origin

# ---------- 全局超参（论文未公开，合理默认【假设】） ----------
ENC_DIM = 16        # 共享编码器输出维度
HID_DIM = 32        # 双分支特征维度
NUM_HEADS = 4       # 跨空间注意力头数
GRID_W, GRID_H = 8, 6   # 检测网格（与 AEGNN 一致）
IMG_W, IMG_H = 240, 180  # N-Caltech101 分辨率
KERNEL_SIZE = 8     # B-spline 核数（AEGNN ncaltech101 同款）
SPLINE_DEGREE = 2   # B-spline 多项式阶数


class EuclideanSplineGCN(nn.Module):
    """欧氏 B-spline GCN 分支（式 1）：h_i' = Σ_j g_Θ(u_ij) h_j。

    多层 SplineConv + BatchNorm + ELU，带残差连接（参考 AEGNN GraphRes）。
    """

    def __init__(self, in_dim: int, hid_dim: int, dim: int = 3,
                 kernel_size: int = KERNEL_SIZE, degree: int = SPLINE_DEGREE,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = SplineConv(in_dim, hid_dim, dim=dim, kernel_size=kernel_size, degree=degree)
        self.norm1 = BatchNorm(hid_dim)
        self.conv2 = SplineConv(hid_dim, hid_dim, dim=dim, kernel_size=kernel_size, degree=degree)
        self.norm2 = BatchNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = self.norm1(h)
        h_res = h
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        h = self.norm2(h)
        h = h + h_res                      # 残差
        return self.dropout(h)


class CrossSpaceAttention(nn.Module):
    """跨空间多头注意力（式 6-8）。

    式 6：Q = W_i^Q·log_o(F^H)，K = W_i^K·F^E，V = W_i^V·F^E
    式 7：Att_i = softmax(QK^T/√d_k)·V
    式 8：Att_out = W^o·Concat(Att_1,...,Att_h)

    双曲特征作为 Query（全局结构语义），欧氏特征作为 Key/Value（局部细节证据），
    在公共切空间做选择性融合。
    """

    def __init__(self, hid_dim: int, num_heads: int = NUM_HEADS, dropout: float = 0.1,
                 chunk_size: int = 2048):
        super().__init__()
        assert hid_dim % num_heads == 0, "hid_dim 必须能被 num_heads 整除"
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads
        self.sqrt_dk = self.head_dim ** 0.5
        self.chunk_size = chunk_size

        self.w_q = nn.Linear(hid_dim, hid_dim, bias=False)
        self.w_k = nn.Linear(hid_dim, hid_dim, bias=False)
        self.w_v = nn.Linear(hid_dim, hid_dim, bias=False)
        self.w_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, f_h: torch.Tensor, f_e: torch.Tensor,
                curvature: torch.Tensor) -> torch.Tensor:
        """f_h: [N, d] 双曲特征；f_e: [N, d] 欧氏特征；返回融合特征 [N, d]。

        curvature 为标量曲率（从双曲分支最后层传入），用于 log 映射。
        f_h 在此处做唯一一次 log 映射（式 6），调用方不要再提前 log。
        分块注意力（chunked softmax）：softmax 按行独立归一化，逐块计算
        在数学上等价于全量 softmax；显存峰值从 O(N²) 降到 O(chunk×N)，
        支持 25000 节点真数据在 8GB 显卡上训练。
        """
        f_ht = poincare_log_origin(f_h, curvature)      # 式 6 前：H → 切空间（唯一一次）
        q = self.w_q(f_ht).view(-1, self.num_heads, self.head_dim)
        k = self.w_k(f_e).view(-1, self.num_heads, self.head_dim)
        v = self.w_v(f_e).view(-1, self.num_heads, self.head_dim)

        # 分块 + 梯度检查点：每个 chunk 的 softmax 中间量在反向时重算，
        # 避免 N 个 chunk 的 s_c/p_c 全部保留在计算图（总激活 ~ O(N²) 会爆 8GB）。
        def _chunk_attn(q_c: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            s_c = torch.einsum("chd,mhd->chm", q_c, k) / self.sqrt_dk   # [C, H, M]
            m_c = s_c.max(dim=-1, keepdim=True).values
            p_c = (s_c - m_c).exp()                                      # [C, H, M]
            l_c = p_c.sum(dim=-1, keepdim=True)
            if self.training:
                p_c = self.dropout(p_c)
            return torch.einsum("chm,mhd->chd", p_c, v) / l_c

        N = q.shape[0]
        out = torch.empty_like(q)
        for c0 in range(0, N, self.chunk_size):
            c1 = min(c0 + self.chunk_size, N)
            if self.training:
                out[c0:c1] = torch.utils.checkpoint.checkpoint(
                    _chunk_attn, q[c0:c1], k, v, use_reentrant=False)
            else:
                out[c0:c1] = _chunk_attn(q[c0:c1], k, v)
        out = out.reshape(-1, self.hid_dim)
        out = self.w_o(out)                                        # 式 8
        return self.norm(out + f_e)                                # 残差 + LayerNorm


class YOLOXHead(nn.Module):
    """YOLOX 解耦检测头（anchor-free，简化版）。

    对每个网格 cell 输出：回归 (x, y, w, h) + 置信度 (1) + 分类 (C)。
    回归目标与 AEGNN 一致：x,y 为相对 cell 左上角的归一化偏移，w,h 为 sqrt 归一化。
    """

    def __init__(self, in_dim: int, num_classes: int,
                 grid: Tuple[int, int] = (GRID_W, GRID_H), dropout: float = 0.1):
        super().__init__()
        self.grid_w, self.grid_h = grid
        self.num_cells = grid[0] * grid[1]
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        # YOLOX 解耦头：每个 cell 独立预测，权重在所有 cell 间共享（卷积式）
        self.reg_head = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),
                                      nn.Linear(in_dim, 4))
        self.obj_head = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),
                                      nn.Linear(in_dim, 1))
        self.cls_head = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),
                                      nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, num_cells, d] 网格特征；返回 [B, grid_h, grid_w, 5+C]。"""
        B = x.shape[0]
        reg = self.reg_head(x)                       # [B, num_cells, 4]
        obj = self.obj_head(x)                       # [B, num_cells, 1]
        cls = self.cls_head(x)                       # [B, num_cells, C]
        out = torch.cat([reg, obj, cls], dim=-1)     # [B, num_cells, 4+1+C]
        return out.view(B, self.grid_h, self.grid_w, 4 + 1 + self.num_classes)


class ECAFGCN(nn.Module):
    """ECAF-GCN 完整模型。"""

    def __init__(self, num_classes: int = 101, enc_dim: int = ENC_DIM, hid_dim: int = HID_DIM,
                 num_heads: int = NUM_HEADS, curvature_init: float = 0.4,
                 dropout: float = 0.1, grid: Tuple[int, int] = (GRID_W, GRID_H),
                 img_size: Tuple[int, int] = (IMG_W, IMG_H)):
        super().__init__()
        self.num_classes = num_classes
        self.grid_w, self.grid_h = grid
        self.img_w, self.img_h = img_size
        self.curvature_init = curvature_init

        # 1) 共享时空编码器：原始节点特征（极性）→ 公共表示
        self.encoder = nn.Sequential()
        self.enc_conv = SplineConv(1, enc_dim, dim=3, kernel_size=KERNEL_SIZE, degree=SPLINE_DEGREE)
        self.enc_norm = BatchNorm(enc_dim)

        # 2) 双分支
        self.euclidean_branch = EuclideanSplineGCN(enc_dim, hid_dim, dropout=dropout)

        # 双曲分支：2 层 HyperbolicGCN（式 3-5），曲率可学习
        self.hyperbolic_branch = nn.ModuleList([
            HyperbolicGraphConv(enc_dim, hid_dim, curvature_init=curvature_init, dropout=dropout),
            HyperbolicGraphConv(hid_dim, hid_dim, curvature_init=curvature_init, dropout=dropout),
        ])

        # 3) 跨空间注意力
        self.cross_attention = CrossSpaceAttention(hid_dim, num_heads=num_heads, dropout=dropout)

        # 4) 融合后投影 + 检测头
        self.fuse = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = YOLOXHead(hid_dim, num_classes=num_classes, grid=grid, dropout=dropout)

    def _last_curvature(self) -> torch.Tensor:
        # 取双曲分支最后一层的曲率（softplus 后的正值）供 log 映射使用
        return self.hyperbolic_branch[-1]._curvature()

    def forward(self, data) -> torch.Tensor:
        """输入 PyG Batch（data.x, data.pos, data.edge_index, data.edge_attr, data.batch）。"""
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        if batch is None:  # 单样本 Data 没有 batch 属性，补全零向量
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1) 共享编码器
        f0 = F.elu(self.enc_conv(x, edge_index, edge_attr))
        f0 = self.enc_norm(f0)

        # 2) 欧氏分支
        f_e = self.euclidean_branch(f0, edge_index, edge_attr)   # [N, hid]

        # 3) 双曲分支：f0 → 指数映射进双曲空间（式 2）→ 双曲 GCN
        c0 = self.hyperbolic_branch[0]._curvature()
        f_h = poincare_exp_origin(f0, c0)                        # 式 2：F^H = exp_o^c(F^E)
        for layer in self.hyperbolic_branch:
            f_h = layer(f_h, edge_index)

        # 4) 跨空间注意力（式 6-8；内部做唯一一次 log 映射）
        f_fused = self.cross_attention(f_h, f_e, self._last_curvature())

        # 5) 网格池化：把节点特征聚合到 8×6 空间网格（每格一个 cell 特征）
        grid_feat = self._grid_pool(f_fused, pos, batch)          # [B, num_cells, hid]

        # 6) 检测头
        grid_feat = self.fuse(grid_feat)
        return self.head(grid_feat)                               # [B, 6, 8, 5+C]

    def _grid_pool(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """把节点特征按空间位置聚合到检测网格（每个 cell 内特征均值）。"""
        cell_w = self.img_w / self.grid_w
        cell_h = self.img_h / self.grid_h
        cx = (pos[:, 0] / cell_w).long().clamp(0, self.grid_w - 1)
        cy = (pos[:, 1] / cell_h).long().clamp(0, self.grid_h - 1)
        n_graphs = int(batch.max().item()) + 1
        gid = batch * (self.grid_w * self.grid_h) + cy * self.grid_w + cx   # 全局 cell id
        feat = scatter(x, gid, dim=0, dim_size=n_graphs * self.grid_w * self.grid_h, reduce="mean")
        return feat.view(n_graphs, self.grid_h * self.grid_w, -1)           # [B, 48, d]

    # ---------- 检测推理辅助 ----------
    def decode(self, model_output: torch.Tensor,
               conf_threshold: float = 0.5) -> torch.Tensor:
        """把网格输出解码为边界框（与 AEGNN detect 语义一致）。

        model_output: [B, grid_h, grid_w, 5+C]
        返回 [num_det, 7]：(batch_idx, x1, y1, x2, y2, cls, obj_conf)
        """
        B, gh, gw, _ = model_output.shape
        device = model_output.device
        cell_w = self.img_w / gw
        cell_h = self.img_h / gh

        reg = model_output[..., 0:4].clamp(min=0)
        obj = torch.sigmoid(model_output[..., 4:5])
        cls_scores = model_output[..., 5:]

        # 网格 cell 左上角坐标
        cell_x = torch.arange(gw, device=device) * cell_w
        cell_y = torch.arange(gh, device=device) * cell_h

        x_rel = reg[..., 0] * cell_w          # 相对 cell 的偏移 → 像素
        y_rel = reg[..., 1] * cell_h
        w = (reg[..., 2] ** 2) * self.img_w   # sqrt 归一化 → 像素
        h = (reg[..., 3] ** 2) * self.img_h

        x1 = cell_x[None, None, :] + x_rel
        y1 = cell_y[None, :, None] + y_rel
        x2 = (x1 + w).clamp(max=self.img_w)
        y2 = (y1 + h).clamp(max=self.img_h)
        x1 = x1.clamp(min=0)
        y1 = y1.clamp(min=0)

        cls_id = cls_scores.argmax(dim=-1)                       # [B, gh, gw]
        cls_conf = cls_scores.gather(-1, cls_id.unsqueeze(-1)).squeeze(-1)  # [B, gh, gw]
        final_conf = obj.squeeze(-1) * cls_conf.sigmoid()         # 综合置信度

        mask = final_conf >= conf_threshold
        idx = mask.nonzero()
        if idx.shape[0] == 0:
            return torch.zeros(0, 7, device=device)

        bi, yi, xi = idx[:, 0], idx[:, 1], idx[:, 2]
        det = torch.stack([
            bi.float(), x1[bi, yi, xi], y1[bi, yi, xi],
            x2[bi, yi, xi], y2[bi, yi, xi],
            cls_id[bi, yi, xi].float(), final_conf[bi, yi, xi],
        ], dim=-1)
        return det
