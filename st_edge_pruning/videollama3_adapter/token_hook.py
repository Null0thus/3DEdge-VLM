from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch

from st_edge_pruning.edge import build_edge_evidence
from st_edge_pruning.pruning.apply_mask import sample_keep_mask, stable_sample_seed, summarize_mask
from st_edge_pruning.pruning.probability import compute_ours_probabilities
from st_edge_pruning.types import PruneConfig


def _as_int(value: Any) -> int:
    """Convert scalar tensors or Python scalars to int."""

    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _video_grid_hw(grid_size: torch.Tensor, merge_size: torch.Tensor) -> Tuple[int, int, int]:
    """Return T, H_p, W_p for VideoLLaMA3 post-merge video tokens."""

    t_size = _as_int(grid_size[0])
    merge = max(_as_int(merge_size), 1)
    hp = _as_int(grid_size[1]) // merge
    wp = _as_int(grid_size[2]) // merge
    return t_size, hp, wp


def _context_frames(context: Optional[Dict[str, Any]], item_index: int):
    """Fetch sampled raw frames used for edge evidence."""

    if not context:
        return None
    frames_by_index = context.get("raw_frames_by_index")
    if isinstance(frames_by_index, dict):
        return frames_by_index.get(item_index)
    return context.get("raw_frames")


def _align_frames(frames: np.ndarray, target_t: int) -> np.ndarray:
    """Match raw-frame count to the visual token time dimension."""

    if frames is None:
        raise ValueError("Ours requires sampled raw video frames for VideoLLaMA3")
    if len(frames) == target_t:
        return frames
    if len(frames) <= 0:
        raise ValueError("Ours received an empty raw-frame array")
    indices = np.linspace(0, len(frames) - 1, target_t, dtype=int)
    return frames[indices]


def _sample_random_mask(shape: Tuple[int, int, int], config: PruneConfig, sample_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a VideoLLaMA3 random-pruning mask with explicit non-square shape."""

    keep_probs = torch.full(shape, float(config.keep_ratio), dtype=torch.float32)
    sample_seed = stable_sample_seed(config.seed, sample_key)
    generator = torch.Generator()
    generator.manual_seed(int(sample_seed))
    random_scores = torch.rand(keep_probs.shape, generator=generator)
    keep_mask = sample_keep_mask(keep_probs, config, scores=random_scores, sample_seed=sample_seed)
    return keep_probs, keep_mask


def _sample_ours_mask(
    shape: Tuple[int, int, int],
    frames: np.ndarray,
    config: PruneConfig,
    sample_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute our edge-evidence probabilities on VideoLLaMA3 token grids."""

    t_size, hp, wp = shape
    frames = _align_frames(frames, t_size)
    evidence = build_edge_evidence(frames, config)
    dummy_features = torch.empty((t_size, hp * wp, 1), dtype=torch.float32)
    keep_probs, importance = compute_ours_probabilities(
        dummy_features,
        evidence,
        config,
        token_grid_hw=(hp, wp),
    )
    sample_seed = stable_sample_seed(config.seed, sample_key)
    keep_mask = sample_keep_mask(keep_probs, config, scores=keep_probs, sample_seed=sample_seed)
    extra_stats = {
        "tau_s": float(evidence["tau_s"].item()),
        "tau_t": float(evidence["tau_t"].item()),
        "dyn_evidence_voxels": int(evidence["gamma_dyn"].sum().item()),
        "sta_evidence_voxels": int(evidence["gamma_sta"].sum().item()),
    }
    return keep_probs, keep_mask, importance, extra_stats


def _record_result(
    context: Optional[Dict[str, Any]],
    item_index: int,
    keep_probs: torch.Tensor,
    keep_mask: torch.Tensor,
    method: str,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Store VideoLLaMA3 compression statistics for the outer eval loop."""

    if context is None:
        return
    stats = summarize_mask(keep_probs, keep_mask)
    stats["method"] = method
    stats["llm_visual_tokens_before"] = int(keep_mask.numel())
    stats["llm_visual_tokens_after"] = int(keep_mask.sum().item())
    stats["llm_visual_keep_ratio"] = float(stats["llm_visual_tokens_after"] / max(stats["llm_visual_tokens_before"], 1))
    stats["model_family"] = "videollama3"
    stats["sampling_mode"] = str(context.get("config", {}).get("sampling_mode", "topk"))
    if extra_stats:
        stats.update(extra_stats)
    context.setdefault("results", {})[item_index] = {
        "keep_probs": keep_probs.detach().cpu(),
        "keep_mask": keep_mask.detach().cpu(),
        "stats": stats,
    }


def _native_difffp_mask(
    model,
    pixel_values: torch.Tensor,
    batched_num_patches: torch.Tensor,
    grid_sizes: torch.Tensor,
    merge_sizes: torch.Tensor,
    modals: List[str],
    threshold: float,
    min_tokens: int,
) -> torch.Tensor:
    """Call the original VideoLLaMA3 DiffFP implementation."""

    return model._st_edge_original_get_compression_mask(
        pixel_values,
        batched_num_patches,
        grid_sizes,
        merge_sizes,
        modals,
        threshold=threshold,
        min_tokens=min_tokens,
    )


def _target_count(num_items: int, keep_ratio: float, rounding: str, min_keep: int) -> int:
    """Compute a bounded top-k budget for one VideoLLaMA3 video item."""

    raw = num_items * keep_ratio
    if rounding == "floor":
        count = math.floor(raw)
    elif rounding == "ceil":
        count = math.ceil(raw)
    else:
        count = round(raw)
    return max(int(min_keep), min(num_items, int(count)))


def _difffp_pixel_diff_grid(
    images: torch.Tensor,
    grid_size: torch.Tensor,
    merge_size: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return native DiffFP adjacent-frame scores on the post-merge token grid."""

    t_size, hp, wp = _video_grid_hw(grid_size, merge_size)
    images = images.view(t_size, hp * wp, -1)
    pixel_diff = torch.abs(images[1:] - images[:-1]).mean(dim=-1) * 255
    # Native DiffFP has no previous frame for t=0, so it forces the first frame
    # above threshold. We keep the same score convention for threshold mode.
    first_frame_scores = torch.full_like(pixel_diff[0:1], float(threshold) + 1.0)
    return torch.cat([first_frame_scores, pixel_diff], dim=0)


def _difffp_topk_mask_for_video(
    images: torch.Tensor,
    grid_size: torch.Tensor,
    merge_size: torch.Tensor,
    config: PruneConfig,
    threshold: float,
    min_tokens: int,
) -> torch.Tensor:
    """Select DiffFP tokens by top-k adjacent-frame difference under a budget."""

    score_grid = _difffp_pixel_diff_grid(images, grid_size, merge_size, threshold)
    t_size, tokens_per_frame = int(score_grid.shape[0]), int(score_grid.shape[1])
    total_tokens = t_size * tokens_per_frame
    mask = torch.zeros_like(score_grid, dtype=torch.bool)

    # The first frame is a visual anchor in native DiffFP because it has no
    # previous frame. Preserving it keeps this top-k variant comparable to the
    # original algorithm while making the remaining budget deterministic.
    mask[0] = True
    frame_floor = max(int(min_tokens), 0)
    if frame_floor > 0 and t_size > 1:
        per_frame_k = min(frame_floor, tokens_per_frame)
        frame_topk = torch.topk(score_grid[1:], k=per_frame_k, dim=1, largest=True).indices
        frame_rows = torch.arange(1, t_size, device=score_grid.device).unsqueeze(1)
        mask[frame_rows, frame_topk] = True

    target = _target_count(total_tokens, float(config.keep_ratio), config.topk_rounding, config.min_keep_tokens)
    # Hard constraints from first-frame and per-frame floors can make the real
    # keep ratio slightly higher than the requested budget; stats record this.
    target = min(total_tokens, max(target, int(mask.sum().item())))
    remaining = target - int(mask.sum().item())
    if remaining > 0:
        flat_scores = score_grid.flatten()
        flat_mask = mask.flatten()
        candidate_scores = flat_scores.masked_fill(flat_mask, -torch.inf)
        chosen = torch.topk(candidate_scores, k=remaining, largest=True).indices
        flat_mask[chosen] = True
    return mask


def _difffp_topk_mask(
    pixel_values: torch.Tensor,
    batched_num_patches: torch.Tensor,
    grid_sizes: torch.Tensor,
    merge_sizes: torch.Tensor,
    modals: List[str],
    config: PruneConfig,
    threshold: float,
    min_tokens: int,
) -> torch.Tensor:
    """Build a full VideoLLaMA3 compression mask with fixed-budget DiffFP top-k."""

    batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
    masks = []
    for images, num_patches, grid_size, merge_size, modal in zip(
        batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
    ):
        if modal == "text":
            masks.append(torch.ones((0,), dtype=torch.bool, device=pixel_values.device))
        elif modal == "image" or _as_int(grid_size[0]) == 1:
            masks.append(torch.ones((_as_int(num_patches),), dtype=torch.bool, device=pixel_values.device))
        elif modal == "video":
            item_mask = _difffp_topk_mask_for_video(images, grid_size, merge_size, config, threshold, min_tokens)
            masks.append(item_mask.flatten())
        else:
            masks.append(torch.ones((0,), dtype=torch.bool, device=pixel_values.device))
    return torch.cat(masks) if masks else torch.ones((0,), dtype=torch.bool, device=pixel_values.device)


def _all_keep_mask(batched_num_patches: torch.Tensor, modals: List[str], device: torch.device) -> torch.Tensor:
    """Keep all visual tokens while preserving VideoLLaMA3 text pseudo-items."""

    masks = []
    for num_patches, modal in zip(batched_num_patches, modals):
        size = 0 if modal == "text" else _as_int(num_patches)
        masks.append(torch.ones((size,), dtype=torch.bool, device=device))
    return torch.cat(masks) if masks else torch.ones((0,), dtype=torch.bool, device=device)


def _expected_mask_len(batched_num_patches: torch.Tensor, modals: List[str]) -> int:
    """Compute the number of visual tokens represented by compression_mask."""

    total = 0
    for num_patches, modal in zip(batched_num_patches, modals):
        if modal != "text":
            total += _as_int(num_patches)
    return total


def _validate_mask(mask: torch.Tensor, batched_num_patches: torch.Tensor, modals: List[str], method: str) -> None:
    """Check that the compression mask exactly matches VideoLLaMA3 tokens."""

    expected = _expected_mask_len(batched_num_patches, modals)
    actual = int(mask.numel())
    if actual != expected:
        raise RuntimeError(f"VideoLLaMA3 {method} mask length mismatch: expected {expected}, got {actual}")


def _patched_get_compression_mask(
    self,
    pixel_values: torch.FloatTensor,
    batched_num_patches: torch.LongTensor,
    grid_sizes: torch.LongTensor,
    merge_sizes: torch.LongTensor,
    modals: List[str],
    threshold: float = 0.1,
    min_tokens: int = 1,
) -> torch.BoolTensor:
    """Return a selectable VideoLLaMA3 compression mask."""

    context = getattr(self, "st_edge_pruning_context", None)
    if context is None or not context.get("enabled", False):
        return _native_difffp_mask(self, pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals, threshold, min_tokens)

    config_dict = context.get("config", {})
    prune_config = context.get("prune_config")
    if not isinstance(prune_config, PruneConfig):
        prune_config = PruneConfig.from_mapping(config_dict)

    method = str(config_dict.get("method", prune_config.method))
    difffp_selection = str(config_dict.get("videollama3_difffp_selection", "threshold"))
    difffp_threshold = float(config_dict.get("videollama3_difffp_threshold", threshold))
    difffp_min_tokens = int(config_dict.get("videollama3_difffp_min_tokens", min_tokens))

    if method == "difffp":
        if difffp_selection == "threshold":
            mask = _native_difffp_mask(
                self,
                pixel_values,
                batched_num_patches,
                grid_sizes,
                merge_sizes,
                modals,
                difffp_threshold,
                difffp_min_tokens,
            )
        elif difffp_selection == "topk":
            mask = _difffp_topk_mask(
                pixel_values,
                batched_num_patches,
                grid_sizes,
                merge_sizes,
                modals,
                prune_config,
                difffp_threshold,
                difffp_min_tokens,
            )
        else:
            raise ValueError(f"Unsupported VideoLLaMA3 DiffFP selection mode: {difffp_selection}")
    elif method == "full":
        mask = _all_keep_mask(batched_num_patches, modals, pixel_values.device)
    elif method in {"random", "ours"}:
        per_item_masks = []
        for item_index, (num_patches, grid_size, merge_size, modal) in enumerate(zip(batched_num_patches, grid_sizes, merge_sizes, modals)):
            if modal == "text":
                per_item_masks.append(torch.ones((0,), dtype=torch.bool, device=pixel_values.device))
                continue
            if modal == "image" or _as_int(grid_size[0]) == 1:
                item_mask = torch.ones((_as_int(num_patches),), dtype=torch.bool, device=pixel_values.device)
                per_item_masks.append(item_mask)
                continue

            shape = _video_grid_hw(grid_size, merge_size)
            expected_item_tokens = _as_int(num_patches)
            actual_item_tokens = shape[0] * shape[1] * shape[2]
            if actual_item_tokens != expected_item_tokens:
                raise RuntimeError(
                    "VideoLLaMA3 token grid mismatch: "
                    f"grid={shape} gives {actual_item_tokens}, num_patches={expected_item_tokens}"
                )
            sample_id = context.get("sample_id", "unknown")
            sample_key = f"videollama3:{method}:{sample_id}:{item_index}"
            if method == "random":
                keep_probs, keep_mask = _sample_random_mask(shape, prune_config, sample_key)
                extra_stats = {}
            else:
                frames = _context_frames(context, item_index)
                keep_probs, keep_mask, _importance, extra_stats = _sample_ours_mask(shape, frames, prune_config, sample_key)

            _record_result(context, item_index, keep_probs, keep_mask, method, extra_stats)
            per_item_masks.append(keep_mask.reshape(-1).to(device=pixel_values.device))

        mask = torch.cat(per_item_masks) if per_item_masks else torch.ones((0,), dtype=torch.bool, device=pixel_values.device)
    else:
        raise ValueError(f"Unsupported VideoLLaMA3 compression method: {method}")

    _validate_mask(mask, batched_num_patches, modals, method)

    # Native DiffFP/full paths do not pass through _record_result above, so we
    # reconstruct per-video stats from the final mask and grid metadata.
    if method in {"difffp", "full"}:
        offset = 0
        for item_index, (num_patches, grid_size, merge_size, modal) in enumerate(zip(batched_num_patches, grid_sizes, merge_sizes, modals)):
            if modal != "video" or _as_int(grid_size[0]) <= 1:
                offset += 0 if modal == "text" else _as_int(num_patches)
                continue
            t_size, hp, wp = _video_grid_hw(grid_size, merge_size)
            count = t_size * hp * wp
            keep_mask = mask[offset : offset + count].detach().cpu().reshape(t_size, hp, wp)
            keep_probs = keep_mask.float()
            extra = {}
            if method == "difffp":
                extra = {
                    "videollama3_difffp_selection": difffp_selection,
                    "videollama3_difffp_threshold": difffp_threshold,
                    "videollama3_difffp_min_tokens": difffp_min_tokens,
                    "target_keep_ratio": float(prune_config.keep_ratio),
                }
            _record_result(context, item_index, keep_probs, keep_mask, method, extra)
            offset += count

    return mask


def register_videollama3_pruning_hook(model) -> None:
    """Patch one loaded VideoLLaMA3 model instance with selectable compression."""

    if hasattr(model, "_st_edge_original_get_compression_mask"):
        return
    model._st_edge_original_get_compression_mask = model._get_compression_mask
    model._get_compression_mask = types.MethodType(_patched_get_compression_mask, model)
