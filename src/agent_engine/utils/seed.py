"""Reproducibility utilities for setting random seeds.

This module provides utilities to set random seeds across different libraries
for reproducible experiments. NumPy and PyTorch are imported lazily in set_seed()
so that config loading (e.g. for SLURM job generation) does not require them.
"""

import os
import random


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch
    - PYTHONHASHSEED environment variable

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy (optional)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch (optional — not available in MLX-only environments)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed_from_env(default: int = 0) -> int:
    """Get seed from PYTHONHASHSEED environment variable.

    Args:
        default: Default seed if environment variable not set

    Returns:
        Seed value
    """
    return int(os.environ.get('PYTHONHASHSEED', default))
