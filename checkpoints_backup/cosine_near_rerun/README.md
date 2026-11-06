# cosine_near 续跑权重备份

## 基本信息

- **对应实验**：secnet_pi2_g16_cosine_near_rerun_b4_baidu
- **测试准确率**：63.60%（当前 G16 系列最好成绩，基线 60.6%）
- **权重类型**：best_checkpoint（EMA 权重，ema=0.999）
- **文件大小**：34,112,392 字节（约 33MB）
- **训练时间**：2026-11-06 16:57
- **随机种子**：42
- **数据集**：N-Caltech101 baidu 小样本（train 3596 / test 934，每样本 10240 点）

## 配置参数

- 远程 16 点 G16，选点方式 cosine_near
- num_point=10240, batch_size=4, grad_accum=8
- 学习率 0.001, AdamW, weight_decay 0.0001
- epoch 300, patience 20, augment 3
- selector_novelty_weight 0.5, learned_M 128, cs_chunk 64

## 说明

该权重从 epoch 23（52.57%）的 checkpoint 断点续跑得到，最终 best 63.60%。
进程在租约到期时被杀（非正常早停），但 best_checkpoint 已完整保存。
