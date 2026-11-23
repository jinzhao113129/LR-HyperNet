"""交付包的配置、随机状态与模型恢复工具。"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RECIPE = dict(num_point=10240, num_category=101, epoch=150, batch_size=4,
              grad_accum=2, learning_rate=.001, decay_rate=.0001,
              optimizer='AdamW', seed=42, augment=0, ema=0., patience=0,
              hyperbolic_gate=True, remote_g16=False, group_norm=False,
              g16_treatment='', sync_bn=True)


def sha256(path: Path) -> str:
    """分块计算文件校验值，避免整份数据进入内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    """原子写入自己的状态文件，防止半写文件被当成完整结果。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def verify_frozen() -> None:
    """校验被冻结的模型文件；有差异立即停止。"""
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    for name, item in manifest.items():
        if sha256(ROOT / name) != item['sha256']:
            raise ValueError(f'模型快照发生变化: {name}')


def training_code_hash() -> str:
    """固定训练/读取实现的联合指纹，续训时拒绝悄悄换代码。"""
    names = ['common.py', 'data_io.py', 'train_ddp.py', 'source_manifest.json']
    content = '\n'.join(f'{name}:{sha256(ROOT / name)}' for name in names)
    return hashlib.sha256(content.encode()).hexdigest()


def seed_all(seed: int) -> None:
    """设置当前进程CPU/CUDA与数据增强随机流。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    """收集当前进程的可恢复随机状态。"""
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state: dict) -> None:
    """恢复当前rank的随机状态，不复用其他rank的CUDA流。"""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def build_model(sync_bn: bool = False, small: bool = False) -> torch.nn.Module:
    """只启用双曲门控；small仅供CPU结构测试，不进入正式训练。"""
    verify_frozen()
    for directory in (ROOT / 'frozen', ROOT / 'frozen/models'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from models.FECNet import FECNet
    kwargs = dict(use_hyperbolic_gate=True)
    if small:
        kwargs['group_number'] = [96, 48, 24, 12]
    model = FECNet(num_classes=101, **kwargs)
    # 原骨干遗留的通道门控/attention在hgate路径不参与前向，显式冻结。
    for name, parameter in model.named_parameters():
        if name.startswith(('attention.', 'group_conv_list.', 'bn_global_list.')):
            parameter.requires_grad_(False)
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """与原训练器一致的0.2标签平滑，不使用PyTorch不同分母的默认公式。"""
    target = torch.full_like(logits, .2 / (logits.shape[-1] - 1))
    target.scatter_(1, labels.reshape(-1, 1), .8)
    return -(target * logits.log_softmax(-1)).sum(-1).mean()


def check_resume(saved: dict, current: dict, world: int, fingerprint: str) -> None:
    """要求同配方、同世界大小、同数据，避免改变实验后悄悄续跑。"""
    if saved['args'] != current or saved['world_size'] != world:
        raise ValueError('续训配置或world_size不同；请使用原命令')
    if saved['data_fingerprint'] != fingerprint:
        raise ValueError('续训数据清单不同')
    if len(saved['rng_states']) != world:
        raise ValueError('checkpoint缺少部分rank的随机状态')
