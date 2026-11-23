"""离线预检查：来源、H5、运行环境与GPU；不启动训练。"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from common import verify_frozen
from data_io import inspect_data


def main() -> None:
    """数据参数可选，--require-four要求确有四张可见CUDA卡。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--require-four', action='store_true')
    args = parser.parse_args()
    verify_frozen()
    result = dict(python=platform.python_version(), system=platform.platform(),
                  architecture=platform.machine(), torch=torch.__version__,
                  cuda_runtime=torch.version.cuda, numpy=np.__version__, h5py=h5py.__version__,
                  cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count())
    if args.require_four:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
            raise ValueError('需要通过CUDA_VISIBLE_DEVICES准确暴露四张GPU')
        result['gpus'] = [torch.cuda.get_device_name(i) for i in range(4)]
        for i in range(4):
            x = torch.ones(4, device=f'cuda:{i}')
            if x.sum().item() != 4:
                raise RuntimeError('CUDA基本运算失败')
        result['topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    if args.data:
        result['data'] = inspect_data(args.data)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
