from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from st_edge_pruning.methods import apply_evs, apply_full, apply_ours, apply_random
from st_edge_pruning.types import PruneConfig, PruneResult


def _select_result_method(method: str):
    """Return the method implementation selected by the experiment config."""

    if method == "full":
        return apply_full
    if method == "random":
        return apply_random
    if method == "evs":
        return apply_evs
    if method == "ours":
        return apply_ours
    raise ValueError(f"Unsupported pruning method: {method}")


def _context_frames(context: Optional[Dict[str, Any]], image_index: int):
    """Fetch raw sampled RGB frames for the current video item."""

    if not context:
        return None
    frames_by_index = context.get("raw_frames_by_index")
    if isinstance(frames_by_index, dict):
        return frames_by_index.get(image_index)
    return context.get("raw_frames")


def _record_result(context: Optional[Dict[str, Any]], image_index: int, result: PruneResult, llm_before: int, llm_after: int) -> None:
    """Store pruning metadata so the outer eval loop can write result files."""

    if context is None:
        return
    result.stats["llm_visual_tokens_before"] = int(llm_before)
    result.stats["llm_visual_tokens_after"] = int(llm_after)
    result.stats["llm_visual_keep_ratio"] = float(llm_after / max(llm_before, 1))
    result.stats["position_encoding"] = str(context.get("config", {}).get("position_encoding", "sequential"))
    context.setdefault("results", {})[image_index] = {
        "keep_probs": result.keep_probs.detach().cpu(),
        "keep_mask": result.keep_mask.detach().cpu(),
        "stats": result.stats,
    }


def _record_sequence_mask(context: Optional[Dict[str, Any]], image_index: int, seq_mask: torch.Tensor, newline_position: str) -> None:
    """Expose original visual sequence positions for position-preserving mode."""

    if context is None:
        return
    # seq_mask is defined over the unpruned visual sequence after LLaVA newline
    # insertion. Position-preserving encoding gathers position ids with the same
    # mask after text/video features are interleaved.
    context.setdefault("visual_sequence_masks", {})[image_index] = seq_mask.detach().cpu()
    context.setdefault("visual_sequence_lengths", {})[image_index] = int(seq_mask.numel())
    context.setdefault("visual_newline_positions", {})[image_index] = newline_position


def _ordinary_flat_mask(keep_mask: torch.Tensor) -> torch.Tensor:
    """Flatten [T,H_p,W_p] ordinary-token mask to [T*Ns]."""

    return keep_mask.reshape(-1).bool()


def _grid_sequence_mask(keep_mask: torch.Tensor, sequence_len: int, device: torch.device) -> torch.Tensor:
    """Map ordinary-token mask to LLaVA grid-newline sequence positions.

    Grid newline tokens are not pruning targets, so they are always kept.
    """

    t_size, hp, wp = keep_mask.shape
    seq_mask = torch.zeros(sequence_len, dtype=torch.bool, device=device)
    flat = keep_mask.to(device=device).bool()
    for t in range(t_size):
        for row in range(hp):
            seq_base = t * hp * (wp + 1) + row * (wp + 1)
            seq_mask[seq_base : seq_base + wp] = flat[t, row]
            seq_mask[seq_base + wp] = True
    return seq_mask


def _frame_sequence_mask(keep_mask: torch.Tensor, sequence_len: int, device: torch.device) -> torch.Tensor:
    """Map ordinary-token mask to frame-newline sequence positions."""

    t_size, hp, wp = keep_mask.shape
    ns = hp * wp
    seq_mask = torch.zeros(sequence_len, dtype=torch.bool, device=device)
    flat = keep_mask.reshape(t_size, ns).to(device=device).bool()
    for t in range(t_size):
        base = t * (ns + 1)
        seq_mask[base : base + ns] = flat[t]
        seq_mask[base + ns] = True
    return seq_mask


def _flat_sequence_mask(keep_mask: torch.Tensor, sequence_len: int, device: torch.device) -> torch.Tensor:
    """Map ordinary-token mask to flat video sequence positions.

    Any tail token beyond T*Ns, such as a single newline token, is preserved.
    """

    flat = _ordinary_flat_mask(keep_mask).to(device=device)
    seq_mask = torch.ones(sequence_len, dtype=torch.bool, device=device)
    seq_mask[: flat.numel()] = flat
    return seq_mask


def _build_sequence_mask(keep_mask: torch.Tensor, sequence_len: int, newline_position: str, device: torch.device) -> torch.Tensor:
    """Create a boolean mask over the actual visual sequence sent to the LLM."""

    if newline_position == "grid":
        return _grid_sequence_mask(keep_mask, sequence_len, device)
    if newline_position == "frame":
        return _frame_sequence_mask(keep_mask, sequence_len, device)
    if newline_position in {"one_token", "no_token"}:
        return _flat_sequence_mask(keep_mask, sequence_len, device)
    raise ValueError(f"Unsupported mm_newline_position: {newline_position}")


def maybe_prune_video_tokens(
    pooled_video_features: torch.Tensor,
    sequence_features: torch.Tensor,
    context: Optional[Dict[str, Any]],
    image_index: int,
    newline_position: str,
) -> torch.Tensor:
    """Prune only pooled ordinary video tokens and return an LLM-ready sequence.

    Text tokens never enter this function. The input `pooled_video_features`
    is H with shape [T, Ns, D] after P and spatial Pool. The input
    `sequence_features` is the visual sequence after LLaVA's newline handling.
    """

    if context is None or not context.get("enabled", False):
        return sequence_features

    config = context.get("prune_config")
    if not isinstance(config, PruneConfig):
        config = PruneConfig.from_mapping(context.get("config", {}))

    method_fn = _select_result_method(config.method)
    frames = _context_frames(context, image_index)
    result = method_fn(pooled_video_features, frames, config, context=context)
    seq_mask = _build_sequence_mask(result.keep_mask, sequence_features.shape[0], newline_position, sequence_features.device)
    pruned_sequence = sequence_features[seq_mask]
    _record_sequence_mask(context, image_index, seq_mask, newline_position)
    _record_result(context, image_index, result, sequence_features.shape[0], pruned_sequence.shape[0])
    return pruned_sequence
