# SECNet N-Caltech101 82.4% 真实协议调研（2026-10-11）

> 来源：Claude 外部渠道调研（arXiv 原文 + GitHub commit 历史 + Semantic Scholar/HuggingFace API），豆包审查确认
> 核心结论：**82.4% 不是单窗准确率，是两阶段协议的产物**

## 一句话结论

官方 82.4% = 先在单个 30ms 滑窗上训练 SECNet（单窗能力上限约 60%），再训练一个跨窗口聚合分类器得到视频级结果。**任何单窗训练/评估都复现不了 82.4%，这不是 bug，是协议不同。**

## 证据链

### 1. 论文原文（arXiv 2412.20803，ICML 2026）

**Table 1 数据集设置**：
- N-Caltech101：Train 20896 / Test 5226 个**窗口**（= 1742 视频 × 3 窗）
- Point Number = **8192**（不是 10240）
- 每视频平均 300ms，切成 30ms 滑窗
- 随机 20% 划分，无官方 split

**Table 9 单窗消融（§4.6.2）**：
> "Ablation study of different Point-based networks in N-Caltech101 with a 300 ms sliding window"
- SECNet 8192点 → **60.04%**
- SECNet 10240点 → **60.60%**

**这是解释所有复现数字的核心证据**：单窗口（哪怕全长 300ms）下 SECNet 只有 ~60%。

**训练超参（§4.1）**：AdamW lr=0.001、cosine、150 epoch；batch size "动态设满 GPU 显存"（RTX 4090 24GB），实际值未给。

### 2. 官方 README 原话

> "For CIFAR10-DVS and N-Caltech101, we first train SECNet on single sliding window, then train a simple classifier to get the results cross a sample."

**两阶段协议的官方确认**。

### 3. 官方 checkpoint 结构（HPC 实测）

- 153 个 `model.*` 前缀的 backbone key（与本地 FECNet 大配置逐 shape 一致）
- 18 个无前缀头 key：`attention(1,1080)`、`classifier 3240→1024→101`
- **3240 = 3 × 1080**（1080 = mean‖max 双池化特征）→ 三个窗口特征拼接后过分类器
- `best_test_acc=82.377` 就是视频级准确率
- **该模型类和训练脚本从未公开**，所以在公开 FECNet 上永远加载不上

### 4. 公开代码不完整实锤

- `train_frequency.py` 第 39 行 `--num_category choices=[10,40]`——参数上就跑不了 101 类
- `models/FECNet.py` 默认小配置 `[32,68,140,284]`，论文要求的大配置 `[64,132,268,540]` 被注释掉
- **第二阶段聚合模型的定义与训练脚本完全缺失**
- 代码 2026-07-29 才一次性放出（5 个 commit，单 main 分支，无 tags/releases）

### 5. 数据生成脚本（generate_ncaltech101.py）

- 3 个固定 30ms 窗：[35,65]/[135,165]/[235,265] ms
- `NUM_POINTS = 10240`（与论文 Table 1 的 8192 矛盾）
- 训练增强：每窗生成 random_crop + reverse_T 两个版本 → train≈62703
- `np.random.shuffle(indices)` 未设 seed → 每次运行划分不同
- mark = 视频索引 → validate() 的按 mark 投票即视频级多数投票

### 6. 全网无第三方复现

- GitHub：0 fork、7 star、仅 1 条 issue（请求发布 HuggingFace 模型，无人回复）
- HuggingFace：0 个模型
- Semantic Scholar：6 篇引用，无一篇报告复现 82.4%
- 综述 arXiv 2606.23078 列出 "SECNet 82.40" 但只是转引

## 百度盘数据的定位

百度盘下载的 train 3596 / test 934（单窗、mark 全唯一）：
- 与论文 Table 1 的 20896/5226 对不上
- 与 generate 脚本输出（train≈62703/test=5226，3窗）也对不上
- **应为第一阶段"单滑窗"数据**（每视频 1 窗，仅保留窗内事件数够 10240 的视频，约占一半）
- test.h5 与 `data_official/test.h5` 逐元素完全相等 → 就是官方分发的测试集
- 生成它的脚本版本未公开【待验证】

## 对我们的影响

### 复现策略修正

| 之前的认知 | 修正后 |
|---|---|
| 单窗跑 54% 是 bug | 单窗上限 ~60%，54% 完全自洽 |
| 要复现 82.4% 需要调参 | 82.4% 是两阶段协议，单窗不可能到 |
| 官方 checkpoint 加载不上是我们的问题 | 官方 checkpoint 是未公开的第二阶段模型，公开代码根本无法加载 |
| 新架构要和 82.4% 比 | 新架构应在**同单窗协议**下和基线比 |

### 逼近 82.4% 的可行路径（基于公开信息推断）

1. 用 official_gen 的 3 窗数据（train 62703 / test 5226）
2. 训窗口级 FECNet（8192 点、AdamW 1e-3、cosine、150ep）
3. 按 mark 多数投票先看视频级（validate() 已支持，是 82.4 的下界近似）
4. 自建第二阶段聚合头（3×1080→1024→101）微调

### 遗留【待验证】

- 真实 batch size（论文只说"动态设满"）
- 8192 点的取样方式（前 8192 还是随机采样）
- 百度盘单窗数据的生成脚本版本
- "filename" split 含义
- 第二阶段训练细节（是否端到端、冻结与否、epoch 数）

## 来源

- [arXiv 2412.20803 (SECNet 论文)](https://arxiv.org/abs/2412.20803) ｜ [全文 HTML v2](https://arxiv.org/html/2412.20803v2)
- [GitHub rhwxmx/SECNet_ICML](https://github.com/rhwxmx/SECNet_ICML) ｜ [README](https://github.com/rhwxmx/SECNet_ICML/blob/main/README.md) ｜ [generate_ncaltech101.py](https://github.com/rhwxmx/SECNet_ICML/blob/main/dataprocess/generate_ncaltech101.py) ｜ [train_frequency.py](https://github.com/rhwxmx/SECNet_ICML/blob/main/train_frequency.py)
- [arXiv 2606.23078（综述，交叉验证 82.40）](https://arxiv.org/abs/2606.23078)
