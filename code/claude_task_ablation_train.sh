#!/bin/bash
# 给 Claude Code 的任务：把消融版本接入训练脚本
cd ~/JiangYouxin/ZhaoJin

/home/jnguan/zhangchenyuan/node/current/bin/claude -p '
在 ~/JiangYouxin/ZhaoJin/ 目录下工作。你之前写好了 hyperbolic_fecnet_ablation.py（HFECNetAblation 类，支持 --ablation none/no_hyperbolic/fixed_curvature/no_centroid）。现在把它接入训练脚本。

## 任务
修改 train_hfecnet.py：
1. 加 --ablation 参数，取值 none/no_hyperbolic/fixed_curvature/no_centroid，默认 none
2. 模型创建处改用 HFECNetAblation（从 hyperbolic_fecnet_ablation.py 导入），传入 ablation 参数
3. 训练日志（tensorboard 和 console）记录当前 ablation 模式
4. 模型保存的文件名带上 ablation 模式（如 best_model_no_hyperbolic.pt），避免不同消融互相覆盖
5. 其他训练逻辑保持不变，加中文注释说明改动点

## 验证
写完后跑一次 CPU 冒烟测试：
~/JiangYouxin/envs/eventprobe/bin/python train_hfecnet.py --ablation no_hyperbolic --use_cpu --epochs 1 --batch_size 8 --num_point 1024

确认能跑通一个 epoch，打印模型名和参数量。如果数据加载太慢就只跑几十个 step，确认训练循环没报错即可。

## 注意
- 不要删除任何现有文件
- 不要修改 hyperbolic_fecnet_ablation.py 本身
- 如果 train_hfecnet.py 里模型创建处有特殊处理（比如加载预训练），保留并适配
' 2>&1
