from __future__ import annotations

import numpy as np
import torch

from st_edge_pruning.edge import build_edge_evidence
from st_edge_pruning.pruning.apply_mask import sample_keep_mask, stable_sample_seed, summarize_mask
from st_edge_pruning.pruning.probability import compute_ours_probabilities
from st_edge_pruning.types import PruneConfig, PruneResult


def apply_ours(video_features: torch.Tensor, frames: np.ndarray, config: PruneConfig, context=None) -> PruneResult:
    """Apply spatio-temporal edge-evidence pruning to pooled video tokens."""

    if frames is None:
        raise ValueError("Ours requires sampled raw video frames in the pruning context")
    evidence = build_edge_evidence(frames, config)
    keep_probs, importance = compute_ours_probabilities(video_features.detach().cpu(), evidence, config)
    # The sample id keeps Bernoulli pruning reproducible while avoiding the same
    # random pattern for different samples with identical token-grid shapes.
    sample_key = f"ours:{context.get('sample_id')}" if context else None
    sample_seed = stable_sample_seed(config.seed, sample_key)
    keep_mask = sample_keep_mask(keep_probs, config, scores=keep_probs, sample_seed=sample_seed)
    stats = summarize_mask(keep_probs, keep_mask)
    stats.update(
        {
            "method": "ours",
            "tau_s": float(evidence["tau_s"].item()),
            "tau_t": float(evidence["tau_t"].item()),
            "dyn_evidence_voxels": int(evidence["gamma_dyn"].sum().item()),
            "sta_evidence_voxels": int(evidence["gamma_sta"].sum().item()),
        }
    )
    return PruneResult(keep_probs=keep_probs, keep_mask=keep_mask, importance=importance, stats=stats)
