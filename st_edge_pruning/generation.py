from __future__ import annotations

from typing import Any, Dict


def build_generation_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build warning-free generation kwargs from experiment config.

    Transformers warns when sampling-only knobs are set during greedy decoding.
    For deterministic evaluation we keep `do_sample=False` and pass neutral
    sampling values, which leaves the decoded sequence unchanged while keeping
    logs readable.
    """

    do_sample = bool(config["do_sample"])
    kwargs: Dict[str, Any] = {
        "do_sample": do_sample,
        "num_beams": int(config["num_beams"]),
        "max_new_tokens": int(config["max_new_tokens"]),
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = float(config["temperature"])
        kwargs["top_p"] = float(config["top_p"])
        if config.get("top_k") is not None:
            kwargs["top_k"] = int(config["top_k"])
    else:
        # These are the neutral GenerationConfig defaults for non-sampling
        # decoding. They override model configs such as top_k=20 that would
        # otherwise emit warnings, without changing greedy/beam-search outputs.
        kwargs["temperature"] = 1.0
        kwargs["top_p"] = 1.0
        kwargs["top_k"] = 50
    return kwargs
