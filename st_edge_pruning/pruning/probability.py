from __future__ import annotations

import math
from typing import Tuple

import torch

from st_edge_pruning.types import PruneConfig
from .token_grid import infer_token_grid, map_evidence_to_token_grid


def compute_importance(e_dyn: torch.Tensor, e_sta: torch.Tensor, config: PruneConfig) -> torch.Tensor:
    """Compute a_{t,j}=exp(beta_dyn*e_dyn + beta_sta*e_sta)."""

    logits = config.beta_dyn * e_dyn + config.beta_sta * e_sta
    # The clamp is configurable and prevents numerical overflow for large
    # working regions while preserving monotonic token ranking.
    logits = logits.clamp(max=config.importance_clip)
    return torch.exp(logits)


def _solve_lambda(scores: torch.Tensor, target_sum: float, pi_min: float, pi_max: float, iters: int, tol: float) -> torch.Tensor:
    """Binary-search the window normalization coefficient lambda."""

    if not (pi_min <= target_sum / max(scores.numel(), 1) <= pi_max):
        raise ValueError("keep_ratio must lie between pi_min and pi_max for clipped normalization")
    low = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
    high = torch.tensor(1.0, dtype=scores.dtype, device=scores.device)
    while torch.clamp(high * scores, pi_min, pi_max).sum().item() < target_sum:
        high = high * 2.0
    for _ in range(iters):
        mid = (low + high) * 0.5
        cur_sum = torch.clamp(mid * scores, pi_min, pi_max).sum()
        if abs(cur_sum.item() - target_sum) <= tol:
            return mid
        if cur_sum.item() < target_sum:
            low = mid
        else:
            high = mid
    return (low + high) * 0.5


def normalize_probs_by_window(scores: torch.Tensor, config: PruneConfig) -> torch.Tensor:
    """Normalize positive token scores to keep probabilities per time window."""

    if not (0.0 < config.keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1]")
    if config.keep_ratio == 1.0:
        return torch.ones_like(scores)
    if not (0.0 <= config.pi_min < config.pi_max <= 1.0):
        raise ValueError("Expected 0 <= pi_min < pi_max <= 1")

    probs = torch.empty_like(scores)
    num_frames = scores.shape[0]
    window_size = max(int(config.window_size), 1)
    flat_scores = scores.view(num_frames, -1)
    flat_probs = probs.view(num_frames, -1)

    for start in range(0, num_frames, window_size):
        end = min(start + window_size, num_frames)
        window_scores = flat_scores[start:end].flatten().clamp_min(1.0e-12)
        # Window-wise scaling preserves all relative importance values but keeps
        # lambda search numerically well-conditioned when importance=exp(logit)
        # becomes very large. Without this, exp(50) can require far more than
        # the default binary-search iterations and overestimates keep_ratio.
        scale = window_scores.max().clamp_min(1.0e-12)
        window_scores = window_scores / scale
        target_sum = config.keep_ratio * window_scores.numel()
        lam = _solve_lambda(window_scores, target_sum, config.pi_min, config.pi_max, config.lambda_solver_iters, config.lambda_solver_tol)
        scaled_scores = flat_scores[start:end].clamp_min(1.0e-12) / scale
        flat_probs[start:end] = torch.clamp(lam * scaled_scores, config.pi_min, config.pi_max)
    return probs


def compute_ours_probabilities(
    video_features: torch.Tensor,
    evidence: dict,
    config: PruneConfig,
    token_grid_hw: Tuple[int, int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Pi_t for each pooled ordinary video token."""

    # LLaVA pooled tokens are square, so the old path infers H_p x W_p. Models
    # such as VideoLLaMA3 use any-resolution grids, so adapters may pass the
    # exact post-merge grid explicitly.
    grid_hw = token_grid_hw or infer_token_grid(video_features)
    e_dyn = map_evidence_to_token_grid(evidence["gamma_dyn"], grid_hw)
    e_sta = map_evidence_to_token_grid(evidence["gamma_sta"], grid_hw)
    importance = compute_importance(e_dyn, e_sta, config)
    keep_probs = normalize_probs_by_window(importance, config)
    return keep_probs, importance
