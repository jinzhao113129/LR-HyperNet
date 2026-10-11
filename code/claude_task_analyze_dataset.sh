#!/bin/bash
# 给 Claude Code 的任务：写数据集分析脚本
cd ~/JiangYouxin/ZhaoJin

/home/jnguan/zhangchenyuan/node/current/bin/claude -p '
请在当前目录 ~/JiangYouxin/ZhaoJin/ 下写一个 Python 脚本 analyze_dataset.py，用于分析 N-Caltech101 事件分类数据集。

## 数据位置
- 训练集: ./data/train.h5
- 测试集: ./data/test.h5
- H5 格式: 包含 data [N, 8192, 4] float32, label [N] int64, mark [N] int64
- data 的 4 个通道: [x, y, t, polarity]（注意：不是 [t,x,y,p]，是 [x,y,t,p]）

## 脚本功能
1. 加载 train.h5 和 test.h5
2. 统计并打印:
   - 训练/测试样本数
   - 类别数（101）
   - 每类样本数（训练集和测试集分别统计）
   - 点数分布（每个样本固定 8192 点，统计有效点数——非零点，但数据可能已经是固定点数）
   - 事件坐标范围: x 的 min/max/mean, y 的 min/max/mean, t 的 min/max/mean
   - 极性分布: polarity=+1 和 polarity=-1（或 0/1）的比例
3. 生成可视化图表（用 matplotlib，Agg 后端）:
   - 类别分布直方图（训练集和测试集并排，前 30 类即可，101 类太密）
   - 每类样本数排序图（降序，显示长尾分布）
   - x 坐标分布直方图
   - y 坐标分布直方图
   - t 坐标分布直方图
   - 极性分布饼图
4. 所有图表保存到 ./eval_results/dataset_analysis/ 目录
5. 统计结果同时保存到 ./eval_results/dataset_analysis/dataset_stats.txt

## 要求
- Python 解释器: ~/JiangYouxin/envs/eventprobe/bin/python（已安装 torch, h5py, numpy, matplotlib）
- 代码加中文注释
- 脚本可以直接运行: ~/JiangYouxin/envs/eventprobe/bin/python analyze_dataset.py
- 不要删除任何文件
- 写完后运行一次验证能跑通，把输出的最后 20 行打印出来

只做这件事，不要做其他的。
' 2>&1
