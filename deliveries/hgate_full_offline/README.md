# N-Caltech101 全量双曲门控：四张3090离线交付

更新：2026-11-23。请先读本页，再搬运文件。当前是**待目标机验收的代码交付版**，不是已验证的四卡成品环境。

本轮固定使用10240点全量数据，训练本地FECNet骨干＋双曲距离门控。不启用G16选点、双曲注意力或其他实验组件。四张3090共同训练一个模型。

论文写8192点，公开生成器和我们HPC已有H5为10240点。本次选择可获得的10240管线继续实验，不等待官方未公开实现。**不把本包称为82.4%的精确复现。** Stage-2是我们自行补充的三窗口方案，也不使用官方82.377%权重初始化。

## 0. 现在能交什么，还缺什么

已提供：冻结模型、四卡训练/续训入口、既有H5登记、原始数据备用处理入口、离线环境构建与安装脚本、Stage-2、CPU测试源码。

尚未提供：朋友机器的系统/驱动信息、据此生成的环境压缩包和完整依赖锁、四卡CUDA实测。模型早期版本有9项CPU测试通过；随后增加了既有H5接入等改动，用户要求暂不测试，所以**最新交付代码未重新测试**。不能把旧测试结果当作本版四卡验证。

开始正式训练前，必须由接收方按本页完成环境检查和短时测速。若暂不允许测试，就停在安装与准备阶段，不直接投150轮。

## 1. 固定协议

| 项目 | 本轮设置 |
|---|---|
| 数据来源 | 优先原样搬运HPC的`ncaltech101_official_gen/train.h5`、`test.h5` |
| 训练／测试 | 62703／5226个窗口，6967／1742条原始记录 |
| 输入布局 | `(t,x,y,p)`，每窗口10240点，101类别 |
| 时间窗口 | 35–65、135–165、235–265ms |
| H5已有增强 | 训练每窗口原始、裁剪、时间反转；测试无增强 |
| 在线增强／EMA／早停 | 全部关闭 |
| 训练种子 | 42；与历史数据划分种子1234分开 |
| 模型宽度／首层分组 | 64→132→268→540／2048 |
| 组件 | 只启用双曲门控，原有傅里叶模块保持 |
| 训练 | 150轮，AdamW，lr0.001，weight decay0.0001，Cosine |
| 四卡 | 每卡batch4，累积2，全局有效batch32，SyncBatchNorm |
| 损失 | 与原训练器一致的0.2标签平滑 |
| 结果 | Stage-1单窗raw准确率；Stage-2另报原始记录级准确率 |

SyncBN一次前向统计覆盖四卡共16条样本，梯度累积后有效batch32。它与单卡batch16/累积2仍可能存在数值差异。正式四卡入口拒绝静默改变batch、累积或卡数。

本地骨干是历史继承的门控重建版，不等同于官方简化发布版。门控的净收益应与同一骨干、数据和配方的内部对照比较；已有旧基线只有在配置完全匹配且训练完整时才复用。本包不自动启动额外基线实验。

## 2. 朋友先回传机器信息（无需联网）

将本目录通过U盘或内网拷到机器后运行：

```bash
bash scripts/collect_machine.sh > machine_info.txt 2>&1
```

回传`machine_info.txt`。需要Linux版本、x86_64架构、glibc、驱动、内存、磁盘和`nvidia-smi topo -m`。只知道“四张3090”不能保证环境包兼容。

不修改驱动，不使用登录口令或私钥打包。若机器由Slurm管理，下面的四卡命令必须放进管理员分配的GPU作业内执行，不能直接在登录节点启动。

## 3. 联网端取得代码

在能访问GitHub的机器上操作，离线端不执行clone：

```bash
git clone --single-branch --branch codex/hgate-full-offline-20261123 https://github.com/jinzhao113129/LR-HyperNet.git
git -C LR-HyperNet rev-parse HEAD > code_commit.txt
```

实际需要的是`LR-HyperNet/deliveries/hgate_full_offline/`整个目录，不能只拷训练入口。`frozen/models/`为已冻结的本地模型快照；`source_manifest.json`记录SHA256。不得用其他机器上同名的FECNet覆盖它。

## 4. 数据主路线：搬运已经生成的全量H5

这次不必重新下载/处理原始数据。优先复用已核查的两个HPC文件，避免改变历史划分和增强。

| HPC路径，相对`/home/jnguan/JiangYouxin/ZhaoJin` | 字节数 |
|---|---:|
| `ncaltech101_official_gen/train.h5` | 10,277,031,928 |
| `ncaltech101_official_gen/test.h5` | 856,552,800 |

HPC访问需要机构网络。密钥仅在自己的电脑使用，不能交给朋友或放进交付包。在PowerShell中下载到一块空间足够的本地盘：

```powershell
New-Item -ItemType Directory -Force D:\hgate_transfer\data | Out-Null
scp -i "$env:USERPROFILE\.ssh\lcr_key" -P 2223 jnguan@10.160.22.46:/home/jnguan/JiangYouxin/ZhaoJin/ncaltech101_official_gen/train.h5 D:\hgate_transfer\data\train.h5
scp -i "$env:USERPROFILE\.ssh\lcr_key" -P 2223 jnguan@10.160.22.46:/home/jnguan/JiangYouxin/ZhaoJin/ncaltech101_official_gen/test.h5 D:\hgate_transfer\data\test.h5
```

下载不是本次已执行步骤。上面的字节数来自只读核查；**大小相同不能代替SHA256**。下载后由来源端与接收端分别计算SHA256比对；HPC全文件校验涉及顺序读取约11GB，按平台要求放到CPU作业/数据传输节点执行，不在登录节点做重I/O。

不要误拷：`ncaltech101_official_baidu`是3596/934轻量单窗；`archive/old_data/data`是8192早期小集，均不是本轮数据。

旧H5只有`data/label/mark`。环境安装好后，运行登记脚本生成旁路清单：

```bash
"$ENV/bin/python" register_existing_h5.py --data "$DATA" --generator-order-confirmed
```

这一步不修改H5，只写`dataset_manifest.json`并计算完整文件SHA。必须确认该H5出自历史生成器：每条训练记录连续9行，顺序为窗口0/1/2，每窗原始/裁剪/反转；测试连续3行。脚本检查行数、连续性和标签，窗口含义仍明确标为根据生成器推导，不伪装成H5原生字段。

旧H5的train和test各自从0编号mark，跨集合重号不说明泄漏；我们没有官方原始文件划分清单，不能仅凭mark证明作者划分。历史数据划分固定NumPy seed1234，但Torch增强随机流未固定，重新生成不保证逐元素复原。因此优先原样搬运，不擅自重生成替换。

## 5. 原始数据备用路线（主路线拿不到才用）

原始事件数据来源：

- 作者页面：https://sites.google.com/site/garrickorchard/datasets/n-caltech101
- Mendeley V1：https://data.mendeley.com/datasets/cy6cvx3ryv/1
- DOI：10.17632/cy6cvx3ryv.1；CC BY 4.0。

下载**事件版**`Caltech101.zip`，不是普通图像Caltech101。作者页面提供镜像。官方生态读取器记录该事件包MD5为`66201824eabb0239c7ab992480b50ba3`，来源：https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/n_caltech101.py 。分类训练不需要边界框annotations。

```bash
python3 verify_raw.py /path/to/Caltech101.zip > raw_archive_check.json
unzip /path/to/Caltech101.zip -d /path/to/raw
```

`verify_raw.py`只用Python标准库，输出实下载字节数、MD5与SHA。公开页面本轮未取得原压缩包的可靠精确字节数，不编造；按上述MD5验收。如果取得的是HPC的`ncaltech101_raw.zip`，已知大小3,992,155,345字节，但尚未确认与作者zip字节相同，不能绕过校验。重新打包文件必须另留来源、文件清单及自己的SHA。

解压后要求`RAW/类别名/*.bin`，101类别。运行：

```bash
"$ENV/bin/python" prepare_data.py --raw-root /path/to/raw/Caltech101 --out /path/to/new_full_10240 --seed 42 --num-points 10240
```

备用生成器按文件名排序，固定seed42做记录级80/20划分，再处理三窗、训练裁剪/反转，流式写H5。输出`split_manifest.json`、`dataset_manifest.json`和明确的skip记录；可用`--split-file`复用之前冻结的完整文件名划分。

**这条备用路线不是旧HPC seed1234数据，也不是作者原始划分。** 不强凑62703/5226；缺窗不得静默补齐。不要把备用生成的数据与旧HPC模型交叉评估后当同协议结果。若产生缺窗，Stage-2会报错，先回收skip清单，由研究负责人决定处理口径。

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
  data/train.h5
  data/test.h5
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

以下示例将整包放到`/srv/hgate_transfer`。允许换目录，但后续变量必须一致：

```bash
TRANSFER=/srv/hgate_transfer
CODE="$TRANSFER/code"
DATA="$TRANSFER/data"
ENV="$HOME/envs/hgate"
RUNS="$HOME/hgate_runs"
cd "$CODE"
bash scripts/install_env_offline.sh "$TRANSFER/env_bundle" "$ENV"
mkdir -p "$RUNS"
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$ENV/bin/python" preflight.py --require-four
"$ENV/bin/python" register_existing_h5.py --data "$DATA" --generator-order-confirmed
"$ENV/bin/python" preflight.py --data "$DATA" --require-four > "$RUNS/preflight.json"
```

安装目标目录必须不存在；安装只使用本地wheel与`--no-index`。登记清单也拒绝覆盖；若搬运包已包含正确的`dataset_manifest.json`，跳过登记，只做校验。

环境安装预期输出`ENV_OK`、四张可见卡；preflight应记录10240点及101类别。发现任何GPU上有他人任务，先协调，脚本不kill进程。`CUDA_VISIBLE_DEVICES`按分配的卡设置，不擅自占用卡。

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

保持不变：150轮、四卡每卡4、累积2、seed42。训练采样按epoch更新，训练尾部不足整批丢弃；测试采样不补齐、不重复，最终计数必须等于5226。只由rank0保存日志与权重。

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

旧H5的窗口与增强含义依据已确认生成器顺序推导；缺窗、重复元数据或标签不一致则停止，不静默丢记录。Stage-2如中断，当前入口需选新目录重新训练头；Stage-1已具备完整续训，不要误用两个入口。

## 12. 出错时回传什么

| 报错 | 处理 |
|---|---|
| pip访问网络／缺wheel | 停止，联网端补包并重新封存，不在离线端临时改版本 |
| GLIBC/驱动/CUDA不兼容 | 回传machine_info和完整报错；重建匹配环境，不升级系统驱动 |
| 原始zip MD5不符 | 核对是否图像版或重新打包版；不跳过来源验证 |
| H5不是62703/5226×10240 | 检查是否拿成轻量/8192旧集，不reshape改样本数 |
| 缺manifest/window字段 | 旧HPC数据先登记；不覆盖H5，不假造原始文件名 |
| NCCL超时／某rank退出 | 回传四卡日志和topo；torchrun结束本作业，勿kill其他用户进程 |
| OOM | 回传完整报错、GPU占用和配置，不中途改正式batch |
| 输出目录存在 | 新实验用新目录；原实验用显式resume；不要删除旧结果 |
| Stage-2指标低 | 保留特征和头权重，回传后分析；不自动换五种方式挑测试最高 |

结果回收：代码commit、`machine_info.txt`、环境版本/锁文件、数据manifest、preflight、两份benchmark、全部metrics/launcher日志、best/last与completed；Stage-2另交source/split/test.json。不要交账号凭据。

复核联系人可直接转述：本次交付为10240自有全量双曲门控，基于现有HPC H5；Stage-2为自建attn_concat。最新代码暂未重测、依赖包待机器信息后构建，不能称四卡已验收。请审查数据接入、SyncBN/DDP、续训和本文操作命令，勿改当前融合快测。
