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

    Sets Python, NumPy, and PyTorch (CPU + CUDA) seeds, plus CuDNN
    deterministic flags. Pin this in every script immediately after
    config parsing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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
