from __future__ import annotations

from collections import deque

import numpy as np
import torch


def filter_3d_components(candidate_mask: torch.Tensor, min_size: int) -> torch.Tensor:
    """Keep connected components with at least min_size voxels.

    Connectivity is the 3D 26-neighborhood over [T,H,W]. This is deliberately
    implemented without scipy so the experiment has fewer environment
    assumptions.
    """

    mask = candidate_mask.detach().cpu().numpy().astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    t_size, h_size, w_size = mask.shape

    offsets = [
        (dt, dy, dx)
        for dt in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dt == 0 and dy == 0 and dx == 0)
    ]

    for t in range(t_size):
        for y in range(h_size):
            for x in range(w_size):
                if visited[t, y, x] or not mask[t, y, x]:
                    continue
                queue = deque([(t, y, x)])
                visited[t, y, x] = True
                component = []
                while queue:
                    cur = queue.popleft()
                    component.append(cur)
                    ct, cy, cx = cur
                    for dt, dy, dx in offsets:
                        nt, ny, nx = ct + dt, cy + dy, cx + dx
                        if nt < 0 or nt >= t_size or ny < 0 or ny >= h_size or nx < 0 or nx >= w_size:
                            continue
                        if visited[nt, ny, nx] or not mask[nt, ny, nx]:
                            continue
                        visited[nt, ny, nx] = True
                        queue.append((nt, ny, nx))
                if len(component) >= min_size:
                    for voxel in component:
                        kept[voxel] = True

    return torch.from_numpy(kept).to(device=candidate_mask.device)
