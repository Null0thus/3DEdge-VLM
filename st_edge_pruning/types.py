from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class EvalSample:
    """Unified sample format used by all dataset loaders."""

    sample_id: str
    dataset: str
    video_path: str
    question: str
    answer: str
    choices: List[str] = field(default_factory=list)
    task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruneConfig:
    """All pruning hyperparameters.

    Every field is filled from config files or command-line arguments. The
    pruning code should not hide experiment-changing constants elsewhere.
    """

    method: str = "full"
    keep_ratio: float = 1.0
    window_size: int = 4
    sampling_mode: str = "topk"
    position_encoding: str = "sequential"
    topk_rounding: str = "round"
    min_keep_tokens: int = 1
    keep_at_least_one_per_frame: bool = True
    work_height: int = 224
    work_width: int = 224
    spatial_sigma: float = 0.8
    spatial_kernel_size: int = 5
    spatial_grad_kernel: str = "sobel"
    temporal_diff: str = "center"
    alpha_s: float = 0.8
    alpha_t: float = 0.7
    min_dyn_cc_size: int = 16
    min_sta_cc_size: int = 16
    beta_dyn: float = 1.0
    beta_sta: float = 0.5
    pi_min: float = 0.02
    pi_max: float = 0.98
    lambda_solver_iters: int = 40
    lambda_solver_tol: float = 1.0e-5
    importance_clip: float = 50.0
    evs_space: str = "embedding"
    evs_metric: str = "cosine"
    evs_anchor_first_frame: bool = True
    seed: int = 42

    @classmethod
    def from_mapping(cls, values: Dict[str, Any]) -> "PruneConfig":
        """Build a config from a larger experiment dictionary."""

        field_names = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in values.items() if k in field_names}
        return cls(**filtered)


@dataclass
class PruneResult:
    """Result returned by a pruning method for ordinary video tokens only."""

    keep_probs: torch.Tensor
    keep_mask: torch.Tensor
    stats: Dict[str, Any]
    importance: Optional[torch.Tensor] = None


@dataclass
class RunPaths:
    """Resolved output paths for one run."""

    output_dir: Path
    predictions_file: Path
    token_stats_file: Path
    timings_file: Path
    skipped_file: Path
    summary_file: Path
    summary_text_file: Path
    args_file: Path
    heatmap_dir: Path
    logs_dir: Path
    run_log_file: Path
