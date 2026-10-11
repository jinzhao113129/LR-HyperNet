import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda version", torch.version.cuda)
    print("gpu count", torch.cuda.device_count())
    print("gpu name", torch.cuda.get_device_name(0))
import h5py
print("h5py", h5py.__version__)
import numpy
print("numpy", numpy.__version__)
try:
    import scipy
    print("scipy", scipy.__version__)
except ImportError:
    print("scipy not installed")
try:
    import tqdm
    print("tqdm", tqdm.__version__)
except ImportError:
    print("tqdm not installed")
