from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_video_frames(
    video_path: str | Path,
    num_frames: int,
    frame_sampling: str = "uniform",
    video_fps: float = 1.0,
    force_sample: bool = True,
) -> Tuple[np.ndarray, str, float]:
    """Load RGB frames using the same broad policy as LLaVA-Video demos."""

    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    avg_fps = float(vr.get_avg_fps())
    video_time = total_frames / max(avg_fps, 1.0e-6)

    if num_frames <= 0:
        frame_idx = [0]
    elif frame_sampling == "fps" and not force_sample:
        step = max(int(round(avg_fps / max(video_fps, 1.0e-6))), 1)
        frame_idx = list(range(0, total_frames, step))
        if len(frame_idx) > num_frames:
            frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    else:
        frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join(f"{idx / max(avg_fps, 1.0e-6):.2f}s" for idx in frame_idx)
    return frames, frame_time, video_time
