"""Unified seeding for reproducible experiments.

Seeds every RNG that matters for this project in one call: Python's ``random``, NumPy,
and (if installed) PyTorch incl. CUDA. Gymnasium environments are seeded separately at
``env.reset(seed=...)`` — see :func:`robotic_arm_trash.rollout.rollout`.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and (if available) PyTorch RNGs. Returns ``seed``."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships via stable-baselines3
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed
