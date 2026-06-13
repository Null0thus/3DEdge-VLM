from __future__ import annotations

from typing import Any, Dict

import inspect
import torch

from .token_hook import register_videollama3_pruning_hook


def _torch_dtype(config: Dict[str, Any]) -> torch.dtype:
    """Map the shared dtype config to a torch dtype."""

    return torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16


def load_videollama3_model(config: Dict[str, Any]):
    """Load VideoLLaMA3 from the local HuggingFace-style model directory."""

    backend = str(config.get("videollama3_backend", "hf_local"))
    if backend != "hf_local":
        raise ValueError(f"Unsupported VideoLLaMA3 backend: {backend}")

    from transformers import AutoModelForCausalLM, AutoProcessor

    device_map = {"": "cuda:0"} if torch.cuda.is_available() and str(config.get("device", "cuda")).startswith("cuda") else None
    load_kwargs = {}
    if bool(config.get("load_8bit", False)):
        load_kwargs["load_in_8bit"] = True
    elif bool(config.get("load_4bit", False)):
        load_kwargs["load_in_4bit"] = True
    print(f"VideoLLaMA3 backend: {backend}")
    print(f"VideoLLaMA3 model path: {config['model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=_torch_dtype(config),
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True)
    print(f"VideoLLaMA3 model class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"VideoLLaMA3 model source: {inspect.getsourcefile(model.__class__) or 'unknown'}")
    print(f"VideoLLaMA3 processor class: {processor.__class__.__module__}.{processor.__class__.__name__}")
    print(f"VideoLLaMA3 use_token_compression: {getattr(model.config, 'use_token_compression', None)}")

    max_visual_tokens = config.get("videollama3_max_visual_tokens")
    if max_visual_tokens is not None and hasattr(processor, "image_processor"):
        # This follows the official VideoLLaMA3 evaluation knob while keeping it
        # controlled by our experiment config.
        processor.image_processor.max_tokens = int(max_visual_tokens)

    register_videollama3_pruning_hook(model)
    tokenizer = processor.tokenizer
    context_len = getattr(model.config, "max_position_embeddings", 32768)
    return tokenizer, model, processor, context_len
