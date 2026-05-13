"""Shared utilities: seeding, logging, paths."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we touch.

    Seeds Python / NumPy / PyTorch (CPU + CUDA) and turns on TF32 +
    cuDNN benchmark for throughput. We do NOT pin cuDNN-deterministic
    or torch.use_deterministic_algorithms: for a 2.1M-param adaptor on
    a frozen encoder, dominant seed variance is from adaptor init and
    sampler order (both seeded here), not cuDNN kernel selection.
    Dropping determinism buys 1.3-2x and enables FlashAttention.

    Pin this in every script immediately after config parsing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for deterministic worker RNGs.

    PyTorch derives a base seed per worker from `torch.initial_seed()`,
    but Python's `random` and NumPy's RNG inside workers aren't reseeded
    automatically — they fork from the parent state. Pass this function
    as `worker_init_fn=seed_worker` on every DataLoader to make any
    in-worker random ops reproducible across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_logger(name: str = "ttic") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def project_root() -> Path:
    """Return the repository root by walking up from this file."""
    return Path(__file__).resolve().parents[2]
