from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_work_video(frames: np.ndarray, work_height: int, work_width: int) -> torch.Tensor:
    """Resize sampled RGB frames to the common working resolution.

    Returned tensor shape is [T, C, H, W] in float32 range [0, 1].
    """

    if frames.ndim != 4:
        raise ValueError(f"Expected frames [T,H,W,C], got shape {frames.shape}")
    tensor = torch.from_numpy(frames).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return F.interpolate(tensor, size=(work_height, work_width), mode="bilinear", align_corners=False)


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel for depth-wise smoothing."""

    if kernel_size <= 1 or sigma <= 0:
        return torch.ones(1, 1, 1, 1, device=device)
    if kernel_size % 2 == 0:
        raise ValueError("spatial_kernel_size must be odd")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1.0e-12)
    return kernel.view(1, 1, kernel_size, kernel_size)


def smooth_video(work_video: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    """Apply light spatial smoothing independently to every color channel."""

    if sigma <= 0 or kernel_size <= 1:
        return work_video
    t, c, h, w = work_video.shape
    kernel = _gaussian_kernel(kernel_size, sigma, work_video.device).repeat(c, 1, 1, 1)
    return F.conv2d(work_video, kernel, padding=kernel_size // 2, groups=c)


def _gradient_kernels(kind: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Sobel or Scharr derivative kernels."""

    if kind == "sobel":
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device) / 8.0
        ky = kx.t()
    elif kind == "scharr":
        kx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=device) / 32.0
        ky = kx.t()
    else:
        raise ValueError(f"Unsupported spatial_grad_kernel: {kind}")
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def compute_spatial_strength(work_video: torch.Tensor, sigma: float, kernel_size: int, grad_kernel: str) -> torch.Tensor:
    """Compute S(x,y,t), the per-frame spatial boundary strength."""

    video = smooth_video(work_video, sigma=sigma, kernel_size=kernel_size)
    t, c, h, w = video.shape
    kx, ky = _gradient_kernels(grad_kernel, video.device)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    fx = F.conv2d(video, kx, padding=1, groups=c)
    fy = F.conv2d(video, ky, padding=1, groups=c)
    return torch.sqrt((fx.square() + fy.square()).sum(dim=1).clamp_min(0.0))


def compute_temporal_strength(work_video: torch.Tensor, temporal_diff: str = "center") -> torch.Tensor:
    """Compute T(x,y,t), the temporal change strength in observed frames."""

    if temporal_diff != "center":
        raise ValueError(f"Unsupported temporal_diff: {temporal_diff}")
    if work_video.shape[0] == 1:
        return torch.zeros(work_video.shape[0], work_video.shape[2], work_video.shape[3], dtype=work_video.dtype)

    diff = torch.zeros_like(work_video)
    diff[1:-1] = (work_video[2:] - work_video[:-2]) * 0.5
    diff[0] = work_video[1] - work_video[0]
    diff[-1] = work_video[-1] - work_video[-2]
    return torch.sqrt(diff.square().sum(dim=1).clamp_min(0.0))
