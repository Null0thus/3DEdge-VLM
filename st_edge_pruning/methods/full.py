from __future__ import annotations

import torch

from st_edge_pruning.pruning.apply_mask import summarize_mask
from st_edge_pruning.pruning.token_grid import infer_token_grid
from st_edge_pruning.types import PruneConfig, PruneResult


def apply_full(video_features: torch.Tensor, frames, config: PruneConfig, context=None) -> PruneResult:
    """Full baseline: keep every pooled ordinary video token."""

    hp, wp = infer_token_grid(video_features)
    keep_probs = torch.ones((video_features.shape[0], hp, wp), dtype=torch.float32)
    keep_mask = torch.ones_like(keep_probs, dtype=torch.bool)
    stats = summarize_mask(keep_probs, keep_mask)
    stats["method"] = "full"
    return PruneResult(keep_probs=keep_probs, keep_mask=keep_mask, stats=stats)
