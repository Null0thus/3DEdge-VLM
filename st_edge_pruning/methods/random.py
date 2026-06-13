from __future__ import annotations

import torch

from st_edge_pruning.pruning.apply_mask import sample_keep_mask, stable_sample_seed, summarize_mask
from st_edge_pruning.pruning.token_grid import infer_token_grid
from st_edge_pruning.types import PruneConfig, PruneResult


def apply_random(video_features: torch.Tensor, frames, config: PruneConfig, context=None) -> PruneResult:
    """Random baseline on pooled ordinary video tokens only."""

    hp, wp = infer_token_grid(video_features)
    keep_probs = torch.full((video_features.shape[0], hp, wp), float(config.keep_ratio), dtype=torch.float32)

    # Random top-k uses random scores but still reports uniform keep probability.
    # The sample-specific seed keeps the baseline reproducible without reusing
    # the same random pattern for every sample.
    sample_key = f"random:{context.get('sample_id')}" if context else None
    sample_seed = stable_sample_seed(config.seed, sample_key)
    generator = torch.Generator()
    generator.manual_seed(int(sample_seed))
    random_scores = torch.rand(keep_probs.shape, generator=generator)
    keep_mask = sample_keep_mask(keep_probs, config, scores=random_scores, sample_seed=sample_seed)
    stats = summarize_mask(keep_probs, keep_mask)
    stats["method"] = "random"
    return PruneResult(keep_probs=keep_probs, keep_mask=keep_mask, stats=stats)
