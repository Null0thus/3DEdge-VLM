from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def ensure_llava_on_path(llava_root: str | Path) -> None:
    """Add third_party/LLaVA-NeXT to sys.path for local imports."""

    root = str(Path(llava_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_llava_video_model(config: Dict[str, Any]):
    """Load the local LLaVA-Video model with command-line controlled overrides."""

    ensure_llava_on_path(config["llava_root"])
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    overwrite_config = {
        "mm_spatial_pool_mode": config["mm_spatial_pool_mode"],
        "mm_spatial_pool_stride": config["mm_spatial_pool_stride"],
        "mm_newline_position": config["mm_newline_position"],
        "mm_patch_merge_type": config["mm_patch_merge_type"],
    }
    model_name = get_model_name_from_path(config["model_path"])
    return load_pretrained_model(
        config["model_path"],
        None,
        model_name,
        load_8bit=bool(config.get("load_8bit", False)),
        load_4bit=bool(config.get("load_4bit", False)),
        torch_dtype=config.get("dtype", "bfloat16"),
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
        overwrite_config=overwrite_config,
    )
