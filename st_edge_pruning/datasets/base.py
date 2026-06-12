from __future__ import annotations

from typing import Any, Dict, List

from st_edge_pruning.types import EvalSample


def build_dataset(config: Dict[str, Any]) -> List[EvalSample]:
    """Create a unified sample list from the selected dataset."""

    name = str(config["dataset"]).lower()
    if name == "mvbench":
        from .mvbench import load_mvbench

        return load_mvbench(config)
    if name in {"videomme", "video-mme", "video_mme"}:
        from .videomme import load_videomme

        return load_videomme(config)
    raise ValueError(f"Unsupported dataset: {name}")
