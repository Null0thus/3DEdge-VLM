from __future__ import annotations

import time

import torch


def cuda_sync(enabled: bool = True) -> None:
    """Synchronize CUDA before measuring wall-clock GPU work."""

    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def now() -> float:
    """Return a high-resolution timestamp."""

    return time.perf_counter()
