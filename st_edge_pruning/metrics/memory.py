from __future__ import annotations

import torch


def get_gpu_memory() -> dict:
    """Return current and peak CUDA memory in MiB."""

    if not torch.cuda.is_available():
        return {"gpu_memory_allocated_mib": None, "gpu_memory_reserved_mib": None, "gpu_memory_peak_mib": None}
    return {
        "gpu_memory_allocated_mib": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu_memory_reserved_mib": torch.cuda.memory_reserved() / (1024 ** 2),
        "gpu_memory_peak_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }
