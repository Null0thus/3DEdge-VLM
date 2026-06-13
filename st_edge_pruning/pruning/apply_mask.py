from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch

from st_edge_pruning.types import PruneConfig


def _target_count(num_items: int, keep_ratio: float, rounding: str, min_keep: int) -> int:
    """Compute exact top-k count for one window."""

    raw = num_items * keep_ratio
    if rounding == "floor":
        count = math.floor(raw)
    elif rounding == "ceil":
        count = math.ceil(raw)
    else:
        count = round(raw)
    return max(min_keep, min(num_items, int(count)))


def stable_sample_seed(base_seed: int, sample_key: Optional[str]) -> int:
    """Create a reproducible per-sample seed for stochastic pruning."""

    if not sample_key:
        return int(base_seed)
    digest = hashlib.sha256(f"{base_seed}:{sample_key}".encode("utf-8")).hexdigest()
    # torch.Generator.manual_seed accepts 64-bit seeds. Keeping 63 bits avoids
    # signed/unsigned surprises across CPU and CUDA generator backends.
    return int(digest[:16], 16) & ((1 << 63) - 1)


def sample_keep_mask(
    keep_probs: torch.Tensor,
    config: PruneConfig,
    scores: Optional[torch.Tensor] = None,
    sample_seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample or select ordinary video tokens from keep probabilities."""

    if config.keep_ratio >= 1.0:
        return torch.ones_like(keep_probs, dtype=torch.bool)

    generator = torch.Generator(device=keep_probs.device)
    generator.manual_seed(int(config.seed if sample_seed is None else sample_seed))
    if config.sampling_mode == "bernoulli":
        mask = torch.rand(keep_probs.shape, generator=generator, device=keep_probs.device) < keep_probs
    elif config.sampling_mode == "topk":
        ranking = scores if scores is not None else keep_probs
        mask = torch.zeros_like(keep_probs, dtype=torch.bool)
        flat_ranking = ranking.view(ranking.shape[0], -1)
        flat_mask = mask.view(mask.shape[0], -1)
        window_size = max(int(config.window_size), 1)
        for start in range(0, ranking.shape[0], window_size):
            end = min(start + window_size, ranking.shape[0])
            window_scores = flat_ranking[start:end].flatten()
            count = _target_count(window_scores.numel(), config.keep_ratio, config.topk_rounding, config.min_keep_tokens)
            chosen = torch.topk(window_scores, k=count, largest=True).indices
            window_mask = flat_mask[start:end].flatten()
            window_mask[chosen] = True
    else:
        raise ValueError(f"Unsupported sampling_mode: {config.sampling_mode}")

    if config.keep_at_least_one_per_frame:
        flat_probs = keep_probs.view(keep_probs.shape[0], -1)
        flat_mask = mask.view(mask.shape[0], -1)
        for frame_idx in range(flat_mask.shape[0]):
            if not flat_mask[frame_idx].any():
                flat_mask[frame_idx, torch.argmax(flat_probs[frame_idx])] = True
    return mask


def summarize_mask(keep_probs: torch.Tensor, keep_mask: torch.Tensor) -> dict:
    """Collect token statistics for ordinary pooled video tokens."""

    before = int(keep_mask.numel())
    after = int(keep_mask.sum().item())
    return {
        "ordinary_video_tokens_before": before,
        "ordinary_video_tokens_after": after,
        "actual_keep_ratio": after / max(before, 1),
        "mean_keep_prob": float(keep_probs.float().mean().item()),
        "min_keep_prob": float(keep_probs.float().min().item()),
        "max_keep_prob": float(keep_probs.float().max().item()),
        "num_frames": int(keep_mask.shape[0]),
        "tokens_per_frame_before": int(keep_mask.shape[1] * keep_mask.shape[2]),
        "token_grid_h": int(keep_mask.shape[1]),
        "token_grid_w": int(keep_mask.shape[2]),
    }
