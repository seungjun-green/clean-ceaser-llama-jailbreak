"""Generic helpers: seeding, device detection, Caesar cipher, logging."""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def caesar_encode(text: str, shift: int = 3) -> str:
    """Caesar-cipher only ASCII letters; preserve case, digits and punctuation."""
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - ord("A") + shift) % 26 + ord("A")))
        else:
            out.append(ch)
    return "".join(out)


def caesar_decode(text: str, shift: int = 3) -> str:
    return caesar_encode(text, shift=-shift)


_LOGGER_INITIALIZED = False


def get_logger(name: str = "caesar_sft", level: int = logging.INFO) -> logging.Logger:
    """Idempotent logger setup with a single stdout handler."""
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    if not _LOGGER_INITIALIZED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _LOGGER_INITIALIZED = True
    return logger


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_param_count(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key, default)
    return val if val else default
