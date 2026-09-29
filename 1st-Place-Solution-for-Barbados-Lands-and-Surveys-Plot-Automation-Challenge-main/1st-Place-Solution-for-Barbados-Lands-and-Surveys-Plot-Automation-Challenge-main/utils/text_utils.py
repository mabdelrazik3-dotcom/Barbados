"""Utility functions for enforcing deterministic text model training."""

import os
import random

import numpy as np
import torch
from transformers import set_seed


def make_deterministic(seed=3407):
    """Make training completely deterministic"""

    # 1. Set all random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)  # HuggingFace transformers seed

    # 2. Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Ensure reproducible algorithms
    torch.use_deterministic_algorithms(True)

    # 4. Handle DataLoader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 5. For some operations that don't have deterministic alternatives
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"✅ Deterministic mode enabled with seed {seed}")
