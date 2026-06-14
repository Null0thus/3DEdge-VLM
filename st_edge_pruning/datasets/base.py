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
    if name in {"nextqa", "next-qa", "next_qa"}:
        from .nextqa import load_nextqa

        return load_nextqa(config)
    if name in {"tempcompass", "temp-compass", "temp_compass"}:
        from .tempcompass import load_tempcompass

        return load_tempcompass(config)
    raise ValueError(f"Unsupported dataset: {name}")
