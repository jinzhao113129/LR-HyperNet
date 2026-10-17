#!/bin/bash
#SBATCH -J sn_g16_eucnear_s42
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 3-00:00:00
#SBATCH --no-requeue
#SBATCH -o eval_results/sn_g16_eucnear_s42_%j.out
#SBATCH -e eval_results/sn_g16_eucnear_s42_%j.err

# 2026-10-15 ai2: G16 六处理训练链（euc_near = 欧氏距离选 16 最近远程点）。
#
# 口径：与 run_full_g16_hypfar_ai2.sh 逐 flag 相同，唯一差异 = --g16_treatment
# （euc_near：RemoteSelector 内部翻 largest=False，选欧氏特征距离最近的
#   非局部 16 点——RP1 原文 Global KNN 方向，历史从未测过）。
# 详细口径说明见 hyp_far 链脚本头部注释（数据/epoch/batch/骨干/profile/时限/点数同源）。
#
# 与 hyp_far 链唯一差异：-J / --log_dir / --log_name / --g16_treatment
# （--augment 2 两条同开——2026-10-17 用户决策"以后跑的所有都做数据增强"）。
cd /home/jnguan/JiangYouxin/ZhaoJin
export PYTHONPATH=/home/jnguan/JiangYouxin/ZhaoJin:$PYTHONPATH
export LD_LIBRARY_PATH=/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16/cudnn8_lib:$LD_LIBRARY_PATH

/home/jnguan/.miniconda/envs/bhli/bin/python train_lr_hyper.py \
    --data_path ./ncaltech101_official_gen/ \
    --num_point 10240 \
    --batch_size 32 \
    --grad_accum 2 \
    --epoch 150 \
    --learning_rate 0.001 \
    --optimizer AdamW \
    --decay_rate 0.0001 \
    --num_category 101 \
    --remote_g 16 \
    --curvature_init 0.4 \
    --log_dir lr_hyper_g16_eucnear_s42_ai2 \
    --log_name /lr_hyper_g16_eucnear_s42_ai2 \
    --g16_profile \
    --g16_profile_p 0.01 \
    --g16_treatment euc_near \
    --augment 2 \
    --seed 42
