from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from st_edge_pruning.types import PruneConfig
from .components import filter_3d_components
from .gradients import build_work_video, compute_spatial_strength, compute_temporal_strength


def build_edge_candidates(spatial_strength: torch.Tensor, temporal_strength: torch.Tensor, alpha_s: float, alpha_t: float) -> Dict[str, torch.Tensor]:
    """Create dynamic and static edge candidates from adaptive quantiles."""

    tau_s = torch.quantile(spatial_strength.flatten(), alpha_s)
    structured = spatial_strength >= tau_s
    temporal_values = temporal_strength[structured]
    tau_t = torch.quantile(temporal_values.flatten(), alpha_t) if temporal_values.numel() > 0 else torch.tensor(0.0)
    dyn = structured & (temporal_strength >= tau_t)
    sta = structured & (temporal_strength < tau_t)
    return {"dyn": dyn, "sta": sta, "tau_s": tau_s, "tau_t": tau_t}


def build_edge_evidence(frames: np.ndarray, config: PruneConfig) -> Dict[str, torch.Tensor]:
    """Build Gamma_dyn and Gamma_sta from sampled raw video frames."""

    work_video = build_work_video(frames, config.work_height, config.work_width)
    spatial = compute_spatial_strength(work_video, config.spatial_sigma, config.spatial_kernel_size, config.spatial_grad_kernel)
    temporal = compute_temporal_strength(work_video, config.temporal_diff)
    candidates = build_edge_candidates(spatial, temporal, config.alpha_s, config.alpha_t)
    gamma_dyn = filter_3d_components(candidates["dyn"], config.min_dyn_cc_size)
    gamma_sta = filter_3d_components(candidates["sta"], config.min_sta_cc_size)
    return {
        "work_video": work_video,
        "spatial": spatial,
        "temporal": temporal,
        "gamma_dyn": gamma_dyn,
        "gamma_sta": gamma_sta,
        "tau_s": candidates["tau_s"],
        "tau_t": candidates["tau_t"],
    }
