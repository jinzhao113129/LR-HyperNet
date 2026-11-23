"""登记HPC既有10240全量H5，仅新建旁路清单，不修改事件文件。"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from common import dump_json, sha256
from data_io import row_coordinates


def describe(path: Path, split: str) -> tuple:
    """读取小型标签/mark并流式计算SHA，不在内存加载11GB事件。"""
    expected = 62703 if split == 'train' else 5226
    with h5py.File(path, 'r') as source:
        if source['data'].shape != (expected, 10240, 4):
            raise ValueError('不是已核查的HPC62703/5226、10240全量数据')
        marks, labels = source['mark'][:].reshape(-1), source['label'][:].reshape(-1)
        if len(labels) != expected or len(marks) != expected:
            raise ValueError('标签或mark长度不同')
        if set(labels.tolist()) != set(range(101)):
            raise ValueError('需要101个类别，标签0..100')
        row_coordinates(source, split, dict(
            window_metadata='inferred_original_crop_reverse_per_window'))
        records = []
        for mark in np.unique(marks):
            found = np.unique(labels[marks == mark])
            if len(found) != 1:
                raise ValueError('同mark有不同标签')
            records.append(dict(id=int(mark), label=int(found[0]), split=split,
                                original_file=None))
    return dict(rows=expected, bytes=path.stat().st_size, sha256=sha256(path)), records


def main() -> None:
    """仅在CPU数据处理机运行；要求明确确认历史窗口行顺序。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--generator-order-confirmed', action='store_true')
    args = parser.parse_args()
    if not args.generator_order_confirmed:
        parser.error('请先确认每mark按窗口0/1/2，每训练窗口按原始/裁剪/反转写入')
    target = args.data / 'dataset_manifest.json'
    if target.exists():
        raise FileExistsError('已有清单，不自动覆盖，请核对来源')
    files, records = {}, []
    for split in ('train', 'test'):
        files[split], found = describe(args.data / f'{split}.h5', split)
        records.extend(found)
        print(f'{split}: rows={files[split]["rows"]} sha256={files[split]["sha256"]}', flush=True)
    dump_json(target, dict(num_points=10240, classes=list(range(101)), records=records,
              files=files, skipped=[], mark_namespace='split_local',
              split_provenance='existing_HPC_10240_generator_numpy_seed1234_not_author_split',
              window_metadata='inferred_original_crop_reverse_per_window',
              raw_split_identity_verified=False, historical_augmentation_torch_seed_fixed=False))
    print('REGISTERED_EXISTING_H5；原始文件名划分不可从局部mark证明，未修改H5', flush=True)


if __name__ == '__main__':
    main()
