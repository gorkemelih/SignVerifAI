"""Reproducibility utilities for SignVerify.

Handles random seed setting for torch, numpy, and python random.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # MPS does not have manual_seed_all equivalent, but setting manual_seed is sufficient
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a seeded torch Generator for DataLoader workers.
    
    Args:
        seed: Optional seed (uses random if None)
    
    Returns:
        Seeded torch Generator
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """
    Worker init function for DataLoader reproducibility.
    
    Use with DataLoader(worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
