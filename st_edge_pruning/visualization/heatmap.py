from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Small dependency-free jet-like colormap for probability maps."""

    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _gray_colormap(values: np.ndarray) -> np.ndarray:
    """Grayscale fallback colormap."""

    g = (np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _to_color(values: np.ndarray, colormap: str) -> np.ndarray:
    """Convert a probability map to RGB."""

    if colormap == "gray":
        return _gray_colormap(values)
    return _jet_colormap(values)


def save_probability_heatmaps(keep_probs, frames: np.ndarray, sample_id: str, output_dir: str | Path, config: Dict[str, Any]) -> None:
    """Save Pi_t as raw values, standalone images, and video-frame overlays."""

    if keep_probs is None:
        return
    sample_dir = Path(output_dir) / "heatmaps" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    probs = keep_probs.detach().cpu().float().numpy()
    max_frames = min(probs.shape[0], frames.shape[0])
    for frame_idx in range(max_frames):
        prob = probs[frame_idx]
        stem = sample_dir / f"frame_{frame_idx:03d}"
        if config.get("save_prob_npy", True):
            np.save(stem.with_name(stem.name + "_prob.npy"), prob)

        frame = Image.fromarray(frames[frame_idx].astype(np.uint8)).convert("RGB")
        heat_small = Image.fromarray(_to_color(prob, config.get("heatmap_colormap", "jet"))).convert("RGB")
        heat = heat_small.resize(frame.size, resample=Image.BILINEAR)

        if config.get("save_prob_png", True):
            heat.save(stem.with_name(stem.name + "_prob.png"))
        if config.get("save_overlay", True):
            alpha = float(config.get("overlay_alpha", 0.45))
            overlay = Image.blend(frame, heat, alpha=alpha)
            overlay.save(stem.with_name(stem.name + "_overlay.png"))
