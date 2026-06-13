from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from st_edge_pruning.pruning.apply_mask import summarize_mask
from st_edge_pruning.pruning.token_grid import infer_token_grid
from st_edge_pruning.types import PruneConfig, PruneResult


def _target_count(num_items: int, keep_ratio: float, rounding: str, min_keep: int) -> int:
    """Compute the deterministic EVS token budget."""

    raw = num_items * keep_ratio
    if rounding == "floor":
        count = math.floor(raw)
    elif rounding == "ceil":
        count = math.ceil(raw)
    else:
        count = round(raw)
    return max(min_keep, min(num_items, int(count)))


def _temporal_change_scores(video_features: torch.Tensor, metric: str) -> torch.Tensor:
    """Score temporal novelty for H[t, j] against H[t-1, j]."""

    t_size, num_spatial, _ = video_features.shape
    scores = torch.zeros((t_size, num_spatial), dtype=torch.float32, device=video_features.device)
    if t_size <= 1:
        return scores

    current = video_features[1:].detach().float()
    previous = video_features[:-1].detach().float()
    if metric == "cosine":
        # EVS-style embedding-space redundancy: high cosine similarity means a
        # token is temporally static, so 1-similarity is the keep score.
        similarity = F.cosine_similarity(current, previous, dim=-1, eps=1.0e-6)
        scores[1:] = (1.0 - similarity).clamp_min(0.0)
    elif metric == "l2":
        scores[1:] = torch.linalg.vector_norm(current - previous, ord=2, dim=-1)
    else:
        raise ValueError(f"Unsupported evs_metric: {metric}")
    return scores


def _select_evs_mask(scores: torch.Tensor, config: PruneConfig) -> torch.Tensor:
    """Select temporally novel tokens while optionally anchoring the first frame."""

    t_size, num_spatial = scores.shape
    total_tokens = scores.numel()
    target_total = _target_count(total_tokens, config.keep_ratio, config.topk_rounding, config.min_keep_tokens)
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    flat_keep = keep_mask.view(-1)
    flat_scores = scores.reshape(-1)

    if bool(config.evs_anchor_first_frame) and t_size > 0:
        # EVS keeps the first frame as a reference anchor. This can make the
        # actual keep ratio higher than requested for extremely short clips.
        keep_mask[0] = True
        remaining_budget = max(target_total - num_spatial, 0)
        candidate_start = num_spatial
    else:
        remaining_budget = target_total
        candidate_start = 0

    candidate_scores = flat_scores[candidate_start:]
    if remaining_budget > 0 and candidate_scores.numel() > 0:
        keep_count = min(int(remaining_budget), candidate_scores.numel())
        chosen = torch.topk(candidate_scores, k=keep_count, largest=True).indices
        flat_keep[candidate_start + chosen] = True
    return keep_mask


def apply_evs(video_features: torch.Tensor, frames, config: PruneConfig, context=None) -> PruneResult:
    """EVS plug-and-play baseline on pooled ordinary video tokens."""

    if config.evs_space != "embedding":
        raise ValueError("Only embedding-space EVS is implemented for pooled video tokens.")

    hp, wp = infer_token_grid(video_features)
    flat_scores = _temporal_change_scores(video_features, config.evs_metric)
    keep_mask = _select_evs_mask(flat_scores, config).reshape(video_features.shape[0], hp, wp)

    # EVS is deterministic. We expose the binary mask as keep_probs so existing
    # heatmap and statistics writers can visualize retained/pruned regions.
    keep_probs = keep_mask.float()
    stats = summarize_mask(keep_probs, keep_mask)
    selected_scores = flat_scores.reshape(-1)[keep_mask.reshape(-1)]
    stats.update(
        {
            "method": "evs",
            "evs_space": config.evs_space,
            "evs_metric": config.evs_metric,
            "evs_anchor_first_frame": bool(config.evs_anchor_first_frame),
            "evs_change_score_mean": float(flat_scores.float().mean().item()),
            "evs_change_score_max": float(flat_scores.float().max().item()),
            "evs_selected_score_min": float(selected_scores.min().item()) if selected_scores.numel() else None,
        }
    )
    return PruneResult(keep_probs=keep_probs, keep_mask=keep_mask, stats=stats, importance=flat_scores.reshape(video_features.shape[0], hp, wp))
