"""H5惰性读取、无填充评估采样和数据契约检查。"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from common import sha256


class EventH5(Dataset):
    """各worker按需独立打开H5，不在每个rank复制10GB数组。"""

    def __init__(self, path: Path):
        """输入校验过的H5路径，缓存长度而不读取全部事件。"""
        self.path, self.handle = str(path), None
        with h5py.File(path, 'r') as source:
            self.length = len(source['label'])

    def __len__(self) -> int:
        """返回窗口条数。"""
        return self.length

    def __getitem__(self, index: int) -> tuple:
        """返回窗口张量、类别与行号，保留评估覆盖检查所需索引。"""
        if self.handle is None:
            self.handle = h5py.File(self.path, 'r')
        x = torch.from_numpy(np.asarray(self.handle['data'][index], dtype=np.float32))
        y = int(np.asarray(self.handle['label'][index]).item())
        return x.transpose(0, 1).contiguous(), y, index

    def __getstate__(self) -> dict:
        """spawn复制时不传递父进程HDF5句柄。"""
        return dict(path=self.path, length=self.length, handle=None)


class ExactEvalSampler(Sampler):
    """按rank切分全部行，不像DistributedSampler那样补齐重复行。"""

    def __init__(self, size: int, rank: int, world: int):
        """输入测试集长度与进程编号，得到互不重叠的索引。"""
        self.indices = range(rank, size, world)

    def __iter__(self):
        """遍历当前rank的真实行。"""
        return iter(self.indices)

    def __len__(self) -> int:
        """返回当前rank的行数。"""
        return len(self.indices)


def inspect_data(directory: Path, full_hash: bool = True) -> dict:
    """按冻结manifest核对H5及元数据，不要求强凑62703/5226。"""
    path = directory / 'dataset_manifest.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta['num_points'] != 10240 or len(meta['classes']) != 101:
        raise ValueError('正式训练要求10240点、101类')
    marks = {}
    for split in ('train', 'test'):
        item, file = meta['files'][split], directory / f'{split}.h5'
        if full_hash and sha256(file) != item['sha256']:
            raise ValueError(f'{split} H5校验失败')
        with h5py.File(file, 'r') as source:
            labels, ids = source['label'][:].reshape(-1), source['mark'][:].reshape(-1)
            if source['data'].shape != (item['rows'], 10240, 4):
                raise ValueError(f'{split}形状不符')
            if len(labels) != item['rows'] or len(ids) != len(labels):
                raise ValueError('标签/mark数量不符')
            if set(labels.tolist()) != set(range(101)):
                raise ValueError(f'{split}未覆盖101类别')
            for start in range(0, len(labels), 128):
                if not np.isfinite(source['data'][start:start + 128]).all():
                    raise ValueError('输入有非有限值')
            check_rows(source, split, meta)
            marks[split] = set(ids.tolist())
    if meta.get('mark_namespace', 'global') == 'global' and marks['train'] & marks['test']:
        raise ValueError('训练测试原始记录泄漏')
    return dict(manifest_sha256=sha256(path), rows={s: meta['files'][s]['rows']
                for s in ('train', 'test')}, split_provenance=meta['split_provenance'],
                raw_split_identity_verified=meta.get('mark_namespace', 'global') == 'global',
                window_metadata=meta.get('window_metadata', 'explicit'))


def row_coordinates(source: h5py.File, split: str, meta: dict) -> tuple:
    """旧H5无窗口字段时，按已确认生成器顺序推导，不修改原H5。"""
    if 'window' in source and 'variant' in source:
        return source['window'][:], source['variant'][:]
    if meta.get('window_metadata') != 'inferred_original_crop_reverse_per_window':
        raise ValueError('H5缺少窗口元数据，且未确认历史生成顺序')
    ids = source['mark'][:].reshape(-1)
    windows, variants = np.zeros(len(ids), dtype='i8'), np.zeros(len(ids), dtype='i8')
    repeats = 9 if split == 'train' else 3
    for mark in np.unique(ids):
        positions = np.flatnonzero(ids == mark)
        if len(positions) != repeats or np.any(np.diff(positions) != 1):
            raise ValueError('旧H5每mark行数或连续性不符合历史生成器，不能推断窗口')
        windows[positions] = np.arange(repeats) // (3 if split == 'train' else 1)
        variants[positions] = np.arange(repeats) % (3 if split == 'train' else 1)
    return windows, variants


def check_rows(source: h5py.File, split: str, meta: dict) -> None:
    """逐mark检查标签、窗口与变体；缺失窗口有显式skip记录而非补数据。"""
    ids, labels = source['mark'][:].reshape(-1), source['label'][:].reshape(-1)
    windows, variants = row_coordinates(source, split, meta)
    rows = list(zip(ids.tolist(), windows.tolist(), variants.tolist()))
    if len(rows) != len(set(rows)):
        raise ValueError('重复mark/window/variant')
    records = {x['id']: x for x in meta['records'] if x['split'] == split}
    for mark in np.unique(ids):
        mask = ids == mark
        if int(mark) not in records or not np.all(labels[mask] == records[int(mark)]['label']):
            raise ValueError('mark标签与原记录清单不一致')
    if not set(windows.tolist()) <= {0, 1, 2}:
        raise ValueError('窗口编号错误')
    allowed = {0, 1, 2} if split == 'train' else {0}
    if not set(variants.tolist()) <= allowed:
        raise ValueError('测试集不应包含增强版本')
    expected = {(r['id'], w, v) for r in records.values() for w in range(3) for v in allowed}
    skipped = {(r['record_id'], r['window'], r['variant']) for r in meta['skipped']
               if r['split'] == split}
    if set(rows) & skipped or set(rows) | skipped != expected:
        raise ValueError('窗口不完整且缺少明确跳过记录')
