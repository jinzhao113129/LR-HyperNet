#!/bin/bash
# 检查 bhli 环境的训练依赖
PY=/home/jnguan/.miniconda/envs/bhli/bin/python
"$PY" -c "import torch, numpy, h5py, sklearn, scipy; print('torch', torch.__version__); print('numpy', numpy.__version__); print('h5py', h5py.__version__); print('sklearn', sklearn.__version__)" 2>&1 | tail -6
echo "=== tensorboard ==="
"$PY" -c "from torch.utils.tensorboard import SummaryWriter; print('tensorboard OK')" 2>&1 | tail -2
echo "=== GPU support check ==="
"$PY" -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>&1 | tail -3
