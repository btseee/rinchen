"""Deterministic seeding across every RNG Rinchen touches."""
from __future__ import annotations

import random


def set_all(seed: int) -> None:
    """Seed Python's random, NumPy, and (if importable) PyTorch."""
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
