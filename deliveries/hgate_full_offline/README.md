# N-Caltech101 全量双曲门控：四张3090离线交付

更新：2026-11-23。请先读本页，再搬运文件。当前是**待目标机验收的代码交付版**，不是已验证的四卡成品环境。

本轮固定使用10240点全量数据，训练本地FECNet骨干＋双曲距离门控。不启用G16选点、双曲注意力或其他实验组件。四张3090共同训练一个模型。

**接收方是自行创建GPU实例的平台，不是Slurm集群。** 流程是创建一台带四张3090的实例，在该实例终端直接运行本页命令；不使用`sbatch`、`srun`或`squeue`，也不需要我们的HPC账号。

论文写8192点，公开生成器和我们HPC已有H5为10240点。本次选择可获得的10240管线继续实验，不等待官方未公开实现。**不把本包称为82.4%的精确复现。** Stage-2是我们自行补充的三窗口方案，也不使用官方82.377%权重初始化。

## 0. 现在能交什么，还缺什么

已提供：冻结模型、四卡训练/续训入口、官方原始数据本地生成入口、离线环境构建与安装脚本、Stage-2、CPU测试源码。主流程不需要HPC账号或现成H5。

尚未提供：朋友机器的系统/驱动信息、据此生成的环境压缩包和完整依赖锁、四卡CUDA实测。模型早期版本有9项CPU测试通过；随后增加了既有H5接入等改动，用户要求暂不测试，所以**最新交付代码未重新测试**。不能把旧测试结果当作本版四卡验证。

开始正式训练前，必须由接收方按本页完成环境检查和短时测速。若暂不允许测试，就停在安装与准备阶段，不直接投150轮。

## 1. 固定协议

| 项目 | 本轮设置 |
|---|---|
| 数据来源 | 接收方自行取得官方原始事件包，在本机生成H5 |
| 训练／测试 | 原始记录80/20划分；62703／5226窗口仅为历史参考，不强凑数量 |
| 输入布局 | `(t,x,y,p)`，每窗口10240点，101类别 |
| 时间窗口 | 35–65、135–165、235–265ms |
| 生成时增强 | 训练每窗口原始、裁剪、时间反转；测试无增强 |
| 在线增强／EMA／早停 | 全部关闭 |
| 种子 | 数据划分/预处理1234；模型训练42 |
| 模型宽度／首层分组 | 64→132→268→540／2048 |
| 组件 | 只启用双曲门控，原有傅里叶模块保持 |
| 训练 | 150轮，AdamW，lr0.001，weight decay0.0001，Cosine |
| 四卡 | 每卡batch4，累积2，全局有效batch32，SyncBatchNorm |
| 损失 | 与原训练器一致的0.2标签平滑 |
| 结果 | Stage-1单窗raw准确率；Stage-2另报原始记录级准确率 |

SyncBN一次前向统计覆盖四卡共16条样本，梯度累积后有效batch32。它与单卡batch16/累积2仍可能存在数值差异。正式四卡入口拒绝静默改变batch、累积或卡数。

本地骨干是历史继承的门控重建版，不等同于官方简化发布版。门控的净收益应与同一骨干、数据和配方的内部对照比较；已有旧基线只有在配置完全匹配且训练完整时才复用。本包不自动启动额外基线实验。

## 2. 创建四卡实例并确认环境

1. 在对方平台创建**同一实例内四张RTX3090**的实例，不是创建四个互不相通的单卡实例。
2. 选择Linux x86_64镜像，优先已有Python3.10和可用PyTorch的镜像。候选依赖为Torch2.3.1/CUDA12.1，但仍需根据实际驱动确认；不要求重装或升级驱动。
3. 确认CPU内存和至少100GB可用磁盘，并询问平台哪些挂载目录是持久化存储。原始包、生成H5、环境和checkpoint都放在可保留的目录，不能默认`/tmp`或系统盘重建后仍在。
4. 创建后进入实例终端，运行`nvidia-smi`，确认四张3090均在同一实例且已分配给本实验。
5. 确认平台是否允许内网/文件上传、实例关机或销毁后是否保留磁盘、是否有空闲回收和最长运行时限。不要照搬我们原租赁平台的回收规则。

有条件时先在CPU环境处理原始数据，再开GPU实例训练，减少等待费用；是否能挂载同一持久盘以对方平台实际能力为准。不安排无意义GPU任务保活。

将本目录通过U盘或内网拷到机器后运行：

```bash
bash scripts/collect_machine.sh > machine_info.txt 2>&1
```

回传`machine_info.txt`。需要Linux版本、x86_64架构、glibc、驱动、内存、磁盘和`nvidia-smi topo -m`。只知道“四张3090”不能保证环境包兼容。

本机信息采集不需要联网，不修改驱动，不包含登录口令或私钥。已有镜像环境如能通过依赖和CUDA检查，可直接使用；缺依赖时走第6节的离线打包，不在离线实例里运行联网pip安装。

## 3. 联网端取得代码

在能访问GitHub的机器上操作，离线端不执行clone：

```bash
git clone --single-branch --branch codex/hgate-full-offline-20261123 https://github.com/jinzhao113129/LR-HyperNet.git
git -C LR-HyperNet rev-parse HEAD > code_commit.txt
```

实际需要的是`LR-HyperNet/deliveries/hgate_full_offline/`整个目录，不能只拷训练入口。`frozen/models/`为已冻结的本地模型快照；`source_manifest.json`记录SHA256。不得用其他机器上同名的FECNet覆盖它。

## 4. 自行取得官方原始事件数据

交付包会随附原始事件包`raw/Caltech101.zip`，接收方不必重复下载，也不搬运我们HPC上的train.h5/test.h5。**H5由接收方在自己的实例中生成**。以下官方链接仅在原始包未随附或校验失败时作为备用来源；GPU实例不能联网时，用另一台联网电脑下载，再通过平台文件上传或内网拷入。

原始事件数据来源：

- 作者页面：https://sites.google.com/site/garrickorchard/datasets/n-caltech101
- Mendeley V1：https://data.mendeley.com/datasets/cy6cvx3ryv/1
- DOI：10.17632/cy6cvx3ryv.1；CC BY 4.0。

进入页面下载事件版 **Caltech101.zip**；作者页面也提供镜像。不要下载普通RGB图像Caltech101，不要下载仅含标注的Caltech101_annotations.zip，也不要用旧8192小集或百度盘3596/934单窗H5代替。

原始事件包已发布MD5：`66201824eabb0239c7ab992480b50ba3`。来源：https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/n_caltech101.py 。本包直接读取作者的5字节bin格式，无需安装spikingjelly或torchvision。

联网端可先检查：

```bash
python3 verify_raw.py /path/to/Caltech101.zip > raw_archive_check.json
```

脚本只依赖标准库，报告实际字节数、MD5及SHA256。不匹配就停止，先核对来源，不跳过校验。下载中的HPC外层包叫`ncaltech101_raw.zip`，里面还套有真正的`Caltech101.zip`；最终交付应提取内层事件包，保留其原始字节并单独校验，不把外层zip改名冒充内层。实际大小与校验值以交付根目录的`raw_source_info.json`为准（数据下载完成后生成）。

## 5. 接收方本地生成10240点全量H5

先按第6–8节安装离线环境，再执行本节；不是要求裸系统直接安装Torch。

准备：原始zip位于`$TRANSFER/raw/Caltech101.zip`；`$DATA`为全新输出目录，不能预先创建。直接在实例终端执行下列命令；预处理使用CPU，不需要Slurm或另起四卡任务。

```bash
bash scripts/prepare_from_official_raw.sh "$TRANSFER/raw/Caltech101.zip" "$DATA" "$ENV"
```

这一条命令会校验原始包、解压、生成H5并检查数据。预期解压结构：

```text
Caltech101/
  accordion/*.bin
  airplanes/*.bin
  ...其余类别目录...
```

若zip目录布局不同，脚本会停止并要求确认目录，不猜测路径。已自行解压时也可以直接运行：

```bash
"$ENV/bin/python" prepare_data.py --raw-root /path/to/Caltech101 --out "$DATA" --seed 1234 --num-points 10240
"$ENV/bin/python" preflight.py --data "$DATA"
```

处理规则：稳定排序原始文件，以数据种子1234固定记录级80/20划分；分别取35–65、135–165、235–265ms三个窗口；按公开生成器的最近端点、毫秒量化去重、下采样/有放回补点和归一化方式处理；训练每窗保留原始、随机裁剪、时间反转三个版本，测试每窗只保留原始版本。

输出`train.h5`、`test.h5`、`split_manifest.json`、`dataset_manifest.json`，明确保存文件名、类别、原始文件SHA、划分、窗口和增强元数据。每个增强动作还单独固定Python/NumPy/Torch随机流；可用`--split-file`复用完整文件名划分。

历史HPC数量为62703／5226，但新生成以实际记录数和skip清单为准，不强凑数量。缺窗明确记录，不能静默补齐。生成失败保留现场，重跑用新输出目录；不覆盖旧H5。

这是按公开处理方式生成的**自有10240全量数据**。作者原始划分未公布；即使使用1234，也不能保证与作者或旧HPC增强数组逐元素相同。论文8192声明与公开10240生成器存在差异，本轮已明确选择10240，不阻塞实验。数据种子1234与训练seed42分开。

`register_existing_h5.py`只保留给已有旧HPC H5的兼容场景；**本次主流程不要运行它**，新生成器已直接写好数据清单和窗口字段。

## 6. 联网Linux机器构建离线环境

建议保留至少100GB可用磁盘，含数据、环境、wheel、日志和权重；这是空间预留，不是实测安装体积。训练逐窗口读取H5，不要求每张卡对应的进程把全部11GB复制进内存。主机内存建议32GB以上，实际先看机器信息。

候选环境：Python3.10.14、PyTorch2.3.1＋CUDA12.1、NumPy1.26.4、h5py3.11.0。该版本组合**待目标驱动确认**；不以`nvidia-smi`显示的CUDA上限代替实际可运行检查。

构建必须在兼容的**Linux x86_64**上，构建机glibc不高于目标机；Windows打出的环境不能搬到Linux。联网构建机需已有conda。参考安装器来源：https://docs.conda.io/projects/miniconda/en/latest/ 。不要修改目标机系统Python。

确认机器信息兼容后才运行：

```bash
TARGET_COMPATIBILITY_CONFIRMED=yes bash scripts/build_env_online.sh /path/to/env_bundle machine_info.txt
```

输出包括可搬迁的Python环境`hgate_py310_cu121.tar.gz`、完整`wheelhouse/`、`requirements.lock`、conda记录及构建版本记录。`build_env/`只用于联网端构建，**不要把它当可直接搬迁的目录**。

构建脚本需联网下载实际依赖，本仓库不包含这些大文件；当前尚未实际构建。不能只有requirements文件就让离线机器安装。PyTorch wheel带运行时依赖，本包无需自行编译CUDA扩展，不需要在目标机安装完整CUDA Toolkit。

参考：
- https://pytorch.org/get-started/previous-versions/
- https://conda.github.io/conda-pack/
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## 7. 搬运目录与校验

整理成以下结构，代码目录取本交付目录的内容：

```text
hgate_transfer/
  code/                       # README、frozen、scripts及全部Python文件
  code_commit.txt
  machine_info.txt
  raw/Caltech101.zip            # 官方原始事件包；不必预先提供H5
  env_bundle/hgate_py310_cu121.tar.gz
  env_bundle/requirements.lock
  env_bundle/wheelhouse/*.whl
  env_bundle/build_versions.txt
```

联网端封存，离线端逐文件校验（标准库即可，不需要Torch）：

```bash
python3 hgate_transfer/code/seal_transfer.py seal /absolute/path/hgate_transfer
# 拷贝后：
python3 /absolute/path/hgate_transfer/code/seal_transfer.py verify /absolute/path/hgate_transfer
```

预期`TRANSFER_OK ... files`。若目标机连Python3都没有，先在联网端对环境压缩包生成`sha256sum`文本；目标端用`sha256sum`校验，再解包环境，用新环境Python运行完整搬运校验。不得把密钥、账号配置、其他实验日志放进搬运目录。

## 8. 离线端安装及数据检查

先选定平台确认会保留的存储目录。以下以`/data/hgate`为示例，**不是对平台挂载点的承诺**；若平台实际路径不同，先改`WORK`。没有该路径权限时不要sudo修改系统，使用平台提供的可写目录。把交付包解压到该目录，形成`$WORK/hgate_transfer/`。

```bash
WORK=/data/hgate
TRANSFER="$WORK/hgate_transfer"
CODE="$TRANSFER/code"
DATA="$WORK/generated/full_10240_seed1234"
ENV="$WORK/envs/hgate"
RUNS="$WORK/runs"
cd "$CODE"
bash scripts/install_env_offline.sh "$TRANSFER/env_bundle" "$ENV"
mkdir -p "$RUNS"
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$ENV/bin/python" preflight.py --require-four
bash scripts/prepare_from_official_raw.sh "$TRANSFER/raw/Caltech101.zip" "$DATA" "$ENV"
"$ENV/bin/python" preflight.py --data "$DATA" --require-four > "$RUNS/preflight.json"
```

安装目标目录必须不存在；安装只使用本地wheel与`--no-index`。数据输出目录也必须不存在；若已成功生成且manifest完整，跳过生成命令，只做校验，不重复解压或覆盖。

环境安装预期输出`ENV_OK`、四张可见卡；preflight应记录10240点及101类别。发现任何GPU上有他人任务，先协调，脚本不kill进程。`CUDA_VISIBLE_DEVICES`按分配的卡设置，不擅自占用卡。

如果平台镜像已有可用环境，先将`ENV`设为该环境根目录，检查`$ENV/bin/python`存在并运行preflight；确认torch、numpy、h5py版本和CUDA后，跳过`install_env_offline.sh`。已有环境与候选环境不同时需记录实际版本，不能假装已使用锁定环境。

## 9. 接收方短时验收（本轮尚未执行）

用户目前要求先准备、不测试。下面是允许开始验收后由朋友执行的命令，不是本次已完成记录。

先单卡测同样一次前向batch16，再四卡每卡4；输出目录各自全新：

```bash
CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 train_ddp.py --mode benchmark --data "$DATA" --out "$RUNS/bench_single" --batch-size 16 --accum 2 --steps 12
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_four_gpu.sh benchmark "$DATA" "$RUNS/bench_four" "$ENV"
"$ENV/bin/python" compare_benchmarks.py "$RUNS/bench_single/benchmark.json" "$RUNS/bench_four/benchmark.json"
```

测速执行真实10240点前向、反传和优化器更新，检查门控投影梯度；只写测速结果，不保存训练权重。包含两批预热，报告每秒样本数、每rank显存峰值。必须先有`gate_gradients_checked=true`、无报错及实际四卡吞吐提升，再安排四卡完整训练。

如果单卡batch16 OOM，不能假装已有公平速度对照；回传报错，由负责人确定更小统计batch的测速口径。不要临时改正式训练batch或开启AMP来“跑通”。四卡DDP需要目标机GPU验证，CPU测试不能替代。

## 10. 正式启动、查看与续训

验收通过后，输出目录必须不存在。这里的`$RUNS`已在前面建立：

```bash
nohup bash scripts/run_four_gpu.sh train "$DATA" "$RUNS/hgate_full_s42" "$ENV" > "$RUNS/hgate_full_s42_launcher.log" 2>&1 < /dev/null &
echo $!
```

保持不变：150轮、四卡每卡4、累积2、seed42。训练采样按epoch更新，训练尾部不足整批丢弃；测试采样不补齐、不重复，最终计数必须等于本次dataset_manifest记录的测试行数（历史参考5226）。只由rank0保存日志与权重。

```bash
tail -n 10 "$RUNS/hgate_full_s42_launcher.log"
tail -n 3 "$RUNS/hgate_full_s42/metrics.jsonl"
nvidia-smi
```

每轮JSON包含单窗raw测试准确率、训练秒数和实际样本数。估时按首个完整epoch的实测时间×剩余epoch，再加验证/保存开销；不沿用3090小数据快测的26分钟估算，不承诺四卡快四倍。

中断后先确认原作业已停止，才在同一个输出目录恢复：

```bash
nohup bash scripts/run_four_gpu.sh resume "$DATA" "$RUNS/hgate_full_s42" "$ENV" > "$RUNS/hgate_full_s42_resume_launcher.log" 2>&1 < /dev/null &
```

保存并恢复raw模型、AdamW、Cosine、已完成epoch、各rank的CPU/CUDA/Python/NumPy随机流及DataLoader随机状态。只恢复完整epoch边界，中断的半轮重跑。拒绝改world size、batch、累积、seed或数据清单后续跑。

`nohup`只能应对终端/SSH断开，**不能保证实例关机、销毁或平台回收后训练继续**。实例重建后，先重新确认持久盘内容、四卡和环境路径，重新设置本页变量，再运行resume。旧进程仍活着时不要启动第二份。停机/到期前回收checkpoint和日志，具体关机、续租和磁盘保留操作按对方平台执行，本包不自动控制实例。

`best_checkpoint.pth`仅作探索记录；`last_checkpoint.pth`为150轮终点主权重。看到`completed.json`才算完整Stage-1完成。不要将中途最高准确率当150轮终局。

独立复评示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 train_ddp.py --mode evaluate --data "$DATA" --out "$RUNS/hgate_full_eval" --checkpoint "$RUNS/hgate_full_s42/last_checkpoint.pth"
```

## 11. Stage-2：我们自己补的三窗口头

本轮采用每窗口1080维特征，窗口注意力加权后拼接为3240维，再分类。**这是我们选择的实现，不是已证实的官方前向。** 历史官方权重探针仅约3%，不把它说成已复现82.377%。不套用官方Stage-2权重。

Stage-1完成后再单独执行，默认只用一张卡，其他卡不自动占用：

```bash
CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" stage2_pipeline.py extract --data "$DATA" --stage1 "$RUNS/hgate_full_s42/last_checkpoint.pth" --out "$RUNS/hgate_features" --batch 8
CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" stage2_pipeline.py train --features "$RUNS/hgate_features" --out "$RUNS/hgate_head"
CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" stage2_pipeline.py evaluate --features "$RUNS/hgate_features" --head "$RUNS/hgate_head/last_head.pth" --out "$RUNS/hgate_stage2_eval"
```

骨干严格恢复并冻结，只接受150轮raw last。提取train/test分别固定seed43/42。Stage-2从零训练100轮，SGD lr0.1、momentum0.9、wd0.0001、Cosine、batch128；按原始记录取10%训练记录作验证，三个增强版本不跨训练/验证。只在最后评估测试集，不用测试成绩挑头。

新生成的H5有显式window/variant字段，Stage-2按字段组合三窗；仅兼容旧H5时才根据历史生成器顺序推导。缺窗、重复元数据或标签不一致则停止，不静默丢记录。Stage-2如中断，当前入口需选新目录重新训练头；Stage-1已具备完整续训，不要误用两个入口。

## 12. 出错时回传什么

| 报错 | 处理 |
|---|---|
| pip访问网络／缺wheel | 停止，联网端补包并重新封存，不在离线端临时改版本 |
| GLIBC/驱动/CUDA不兼容 | 回传machine_info和完整报错；重建匹配环境，不升级系统驱动 |
| 原始zip MD5不符 | 核对是否图像版或重新打包版；不跳过来源验证 |
| H5不符合manifest或不是10240点 | 检查原始数据来源和skip清单，不强凑62703/5226，不reshape改样本数 |
| 缺manifest/window字段 | 本次应由prepare_data.py生成；回查生成日志，不拿轻量H5混用 |
| NCCL超时／某rank退出 | 回传四卡日志和topo；torchrun结束本作业，勿kill其他用户进程 |
| OOM | 回传完整报错、GPU占用和配置，不中途改正式batch |
| 输出目录存在 | 新实验用新目录；原实验用显式resume；不要删除旧结果 |
| Stage-2指标低 | 保留特征和头权重，回传后分析；不自动换五种方式挑测试最高 |

结果回收：代码commit、`machine_info.txt`、环境版本/锁文件、数据manifest、preflight、两份benchmark、全部metrics/launcher日志、best/last与completed；Stage-2另交source/split/test.json。不要交账号凭据。

复核联系人可直接转述：本次交付为10240自有全量双曲门控，由接收方取得官方原始事件数据并生成H5；Stage-2为自建attn_concat。最新代码暂未重测、依赖包待机器信息后构建，不能称四卡已验收。请审查数据接入、SyncBN/DDP、续训和本文操作命令，勿改当前融合快测。
