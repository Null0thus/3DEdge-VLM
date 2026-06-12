from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def infer_token_grid(video_features: torch.Tensor) -> Tuple[int, int]:
    """Infer the pooled spatial grid H_p x W_p from [T,Ns,D] features."""

    num_tokens = int(video_features.shape[1])
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected square pooled token grid, got Ns={num_tokens}")
    return side, side


def map_evidence_to_token_grid(evidence_mask: torch.Tensor, token_grid_hw: Tuple[int, int]) -> torch.Tensor:
    """Aggregate voxel evidence counts into the pooled token grid."""

    hp, wp = token_grid_hw
    evidence = evidence_mask.float().unsqueeze(1)
    pooled_ratio = F.adaptive_avg_pool2d(evidence, output_size=(hp, wp)).squeeze(1)
    region_area = evidence_mask.shape[-2] * evidence_mask.shape[-1] / float(hp * wp)
    return pooled_ratio * region_area
