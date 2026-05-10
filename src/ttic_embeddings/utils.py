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

    Sets Python, NumPy, and PyTorch (CPU + CUDA) seeds, CuDNN
    deterministic flags, and PyTorch's global deterministic-algorithms
    switch. CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first CUDA
    init for cublas determinism, so we set it here unconditionally.

    `warn_only=True` on use_deterministic_algorithms because some ops
    (e.g. CUDA scatter_add) lack deterministic implementations; we want
    a warning so the user can decide, not a crash. The encoder-comparison
    effect-size analysis is the main reason this matters — silent
    nondeterminism would inflate seed variance.

    Pin this in every script immediately after config parsing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


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
