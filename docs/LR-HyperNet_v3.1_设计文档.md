# LR-HyperNet v3.1 改进设计文档

> 基于 2026-10-10 / 10-11 论文阅读的可落地用法，筛选 7 项有效改进
> 实施原则：先修稳定性问题，再提性能，最后架构升级
> 基线：v3（KDA 时间编码 + 残差连接 + beta上限 + state范数裁剪）

---

## 第一批：修稳定性（必须做，不依赖 v3 结果）

### 1. 曲率优化三件套（CurvatureAwareOpt, 2024）

**问题**：审查报告 L3——curvature 无上界（`abs+1e-4` 只保正），学到很大时 Poincaré 球半径 1/√c→0、特征挤向原点、注意力退化为均匀。

**改动**：
- **优化器分组**：曲率参数（`hyper_agg.curvature`）单独 param group，学习率 = 主 lr × 0.01
- **曲率正则化**：损失加 `L_c = λ_c · (c - c_init)²`，λ_c = 0.01，防止曲率偏离初始值过远
- **Warmup**：前 10 epoch 固定曲率（`requires_grad=False`），第 11 epoch 起放开学习

**文件**：`train_lr_hyper.py`（优化器 + 损失 + 训练循环）
**验收**：TensorBoard Curvature 曲线稳定在 [0.3, 0.5] 区间，不再异常增大

### 2. Lorentz 模型替代 Poincaré（HyperbolicPointCloudCL, 2025）

**问题**：审查报告 L1——Poincaré 球边界 atanh 梯度尖峰（d(atanh)/d(arg)≈5×10⁵），特征范数大时触边界。

**改动**：
- 新增 `lorentz_exp_origin` / `lorentz_log_origin` / `lorentz_distance` 函数
- `HyperbolicRemoteAggregator` 新增 `--hyperbolic_type lorentz` 选项
- Lorentz 模型：特征表示为 (x₀, x₁,...,x_d)，满足 x₀² - Σxᵢ² = 1/c
- 指数映射：Lorentz 指数映射（无边界问题）
- 距离：Lorentz 双曲余弦距离 `d(x,y) = (1/√c)·arccosh(c·⟨x,y⟩_L)`

**文件**：`models/local_remote_grouper.py`（新增 Lorentz 运算 + HyperbolicRemoteAggregator 选项）
**验收**：前向/反向无 NaN，梯度范数稳定（无尖峰），与 Poincaré 版本精度可比

---

## 第二批：提性能（v3 验证有效后做）

### 3. KDA 双向因果递推（HydraMamba, 2025）

**问题**：KDA 当前是单向因果递推（只看过去），事件分类是离线任务，浪费未来信息。

**改动**：
- `KDACausalEncoder.forward` 新增 `bidirectional` 选项
- 前向 KDA（时间正序）+ 后向 KDA（时间逆序），输出拼接后 Linear 融合
- 新增 `--kda_bidirectional` flag

**文件**：`models/local_remote_grouper.py`（KDACausalEncoder）
**验收**：精度高于单向版本，计算量翻倍但可接受

### 4. 分离式时空嵌入（Event2Vec, ICML 2026）

**问题**：当前 `Linear(4→64)` 对 (t,x,y,p) 耦合编码，空间结构和时间结构被混合。

**改动**：
- x,y：可学习空间查找表（分辨率 240×180，输出 32 维）
- t：1D 卷积时间嵌入（kernel=3，输出 16 维）
- p：可学习极性嵌入（2 类，输出 16 维）
- 三者拼接 → 64 维，与原输出维度一致
- 新增 `--embedding_type separated` flag

**文件**：`lr_hypernet.py`（embed_dim 替换）
**验收**：参数量变化 < 5%，精度不低于原 Linear 版本

### 5. KDA α 事件密度自适应（PASS, NeurIPS 2025）

**问题**：KDA 的 α 遗忘门全局共享，不同事件密度的样本（运动快/慢）用同一遗忘策略。

**改动**：
- α 的 MLP 输入从 `concat([x, state_norm])` 改为 `concat([x, state_norm, event_density])`
- event_density = 组内事件数 / 时间跨度（归一化到 [0,1]）
- 高密度组 α 更小（保留更久），低密度组 α 更大（快速遗忘）

**文件**：`models/local_remote_grouper.py`（KDACausalEncoder 的 alpha_proj 输入）
**验收**：不同密度样本的 α 值有差异，精度稳定

---

## 第三批：架构升级（第二批验证有效后做）

### 6. 全程双曲 MLP（HyperbolicViT, 2022）

**问题**：当前双曲注意力算完立刻 exp 映射回欧氏做 MLP，双曲几何的层级建模优势可能被欧氏 MLP 破坏。

**改动**：
- 新增 `MobiusLinear`（Möbius 矩阵乘法）和 `mobius_relu`（双曲激活）
- 远程聚合后的特征变换从欧氏 MLP 换成双曲 MLP
- 直到分类前才投影回欧氏
- 新增 `--hyperbolic_mlp` flag

**文件**：`models/local_remote_grouper.py`（新增双曲 MLP + HyperbolicRemoteAggregator 输出层）
**验收**：全程双曲，分类前投影，精度稳定

### 7. 双曲对比损失（HyperbolicPointCloudCL, 2025）

**问题**：N-Caltech101 只有 3596 训练样本，从零训练容易过拟合。

**改动**：
- 数据增强：同一事件样本的不同时间窗口 / 随机裁剪 / 翻转作为正样本对
- 双曲对比损失：正样本对在双曲空间拉近、负样本对推远
- 损失 = 交叉熵 + λ_cl · 双曲 InfoNCE，λ_cl = 0.1
- 新增 `--contrastive_loss` flag

**文件**：`train_lr_hyper.py`（数据增强 + 损失函数）+ `provider_data.py`（增强）
**验收**：训练集精度与测试集精度差距缩小（过拟合缓解）

---

## 实施顺序

| 批次 | 改动 | 依赖 | 预计工时 |
|------|------|------|----------|
| 1 | 曲率优化三件套 | 无 | 0.5h |
| 1 | Lorentz 模型 | 无 | 4h |
| 2 | KDA 双向 | v3 稳定 | 1h |
| 2 | 分离式嵌入 | v3 稳定 | 1h |
| 2 | α 密度自适应 | v3 稳定 | 1h |
| 3 | 全程双曲 MLP | 第二批有效 | 3h |
| 3 | 双曲对比损失 | 第二批有效 | 4h |

## 验证策略

- 每批改动先跑 5 epoch smoke（百度数据，b8），确认无 NaN/崩溃
- 再跑 30 epoch 小版本，看收敛曲线
- 最后跑 150 epoch 正式版与基线对比
- 每次只改一个变量，消融实验明确各改动贡献

---

*维护：豆包 | 创建：2026-10-11*
