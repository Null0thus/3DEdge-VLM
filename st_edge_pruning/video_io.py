from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clip_time_bounds(total_frames: int, fps: float, start_time: float | None, end_time: float | None) -> Tuple[int, int]:
    """Convert optional second-based bounds into a non-empty frame interval."""

    safe_fps = max(float(fps), 1.0e-6)
    start_idx = int(np.floor(float(start_time) * safe_fps)) if start_time is not None else 0
    end_idx = int(np.ceil(float(end_time) * safe_fps)) if end_time is not None else total_frames
    start_idx = min(max(start_idx, 0), max(total_frames - 1, 0))
    end_idx = min(max(end_idx, start_idx + 1), total_frames)
    return start_idx, end_idx


def _sample_frame_indices(
    total_frames: int,
    fps: float,
    num_frames: int,
    frame_sampling: str,
    video_fps: float,
    force_sample: bool,
) -> List[int]:
    """Choose frame indices inside a clipped interval."""

    if total_frames <= 0:
        raise ValueError("Cannot sample frames from an empty video or frame directory.")
    if num_frames <= 0:
        return [0]

    if frame_sampling == "fps" and not force_sample:
        # FPS sampling keeps the requested temporal density and caps overly long
        # clips to num_frames, matching the LLaVA demo behavior.
        step = max(int(round(float(fps) / max(float(video_fps), 1.0e-6))), 1)
        frame_idx = list(range(0, total_frames, step))
        if len(frame_idx) > num_frames:
            frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        return frame_idx

    # Uniform sampling keeps exactly num_frames positions; duplicates are allowed
    # when a very short clip has fewer frames than requested.
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def _load_frame_directory(frame_dir: Path, frame_indices: List[int]) -> np.ndarray:
    """Load RGB frames from an MVBench frame directory."""

    from PIL import Image

    frame_files = sorted(path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES)
    if not frame_files:
        raise FileNotFoundError(f"No image frames found in directory: {frame_dir}")
    images = []
    for index in frame_indices:
        # Indices are already clipped by the caller; this guard keeps the loader
        # robust if future datasets provide unusual timing metadata.
        safe_index = min(max(index, 0), len(frame_files) - 1)
        with Image.open(frame_files[safe_index]) as image:
            images.append(np.asarray(image.convert("RGB")))
    return np.stack(images, axis=0)


def load_video_frames(
    video_path: str | Path,
    num_frames: int,
    frame_sampling: str = "uniform",
    video_fps: float = 1.0,
    force_sample: bool = True,
    start_time: float | None = None,
    end_time: float | None = None,
    frame_dir_fps: float = 3.0,
) -> Tuple[np.ndarray, str, float]:
    """Load RGB frames from a video file or an extracted-frame directory."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    if video_path.is_dir():
        frame_files = sorted(path for path in video_path.iterdir() if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES)
        if not frame_files:
            raise FileNotFoundError(f"No image frames found in directory: {video_path}")
        avg_fps = max(float(frame_dir_fps), 1.0e-6)
        start_idx, end_idx = _clip_time_bounds(len(frame_files), avg_fps, start_time, end_time)
        local_count = end_idx - start_idx
        local_indices = _sample_frame_indices(local_count, avg_fps, num_frames, frame_sampling, video_fps, force_sample)
        frame_idx = [start_idx + idx for idx in local_indices]
        frames = _load_frame_directory(video_path, frame_idx)
        video_time = len(frame_files) / avg_fps
    else:
        from decord import VideoReader, cpu

        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        avg_fps = float(vr.get_avg_fps())
        start_idx, end_idx = _clip_time_bounds(total_frames, avg_fps, start_time, end_time)
        local_count = end_idx - start_idx
        local_indices = _sample_frame_indices(local_count, avg_fps, num_frames, frame_sampling, video_fps, force_sample)
        frame_idx = [start_idx + idx for idx in local_indices]
        frames = vr.get_batch(frame_idx).asnumpy()
        video_time = total_frames / max(avg_fps, 1.0e-6)

    frame_time = ",".join(f"{idx / max(avg_fps, 1.0e-6):.2f}s" for idx in frame_idx)
    return frames, frame_time, video_time
