from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _avg(records: List[Dict[str, Any]], key: str):
    values = [r[key] for r in records if r.get(key) is not None]
    return mean(values) if values else None


def write_summary(path: str | Path, predictions: List[Dict[str, Any]], token_stats: List[Dict[str, Any]], timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Write aggregate metrics for one run."""

    total = len(predictions)
    correct = sum(1 for r in predictions if r.get("correct"))
    summary = {
        "num_samples": total,
        "accuracy": correct / max(total, 1),
        "avg_actual_keep_ratio": _avg(token_stats, "actual_keep_ratio"),
        "avg_ordinary_video_tokens_before": _avg(token_stats, "ordinary_video_tokens_before"),
        "avg_ordinary_video_tokens_after": _avg(token_stats, "ordinary_video_tokens_after"),
        "avg_llm_visual_tokens_before": _avg(token_stats, "llm_visual_tokens_before"),
        "avg_llm_visual_tokens_after": _avg(token_stats, "llm_visual_tokens_after"),
        "avg_total_inference_time_sec": _avg(timings, "total_inference_time_sec"),
        "avg_llm_prefill_time_sec": _avg(timings, "llm_prefill_time_sec"),
        "max_gpu_memory_peak_mib": max([r.get("gpu_memory_peak_mib") or 0 for r in timings], default=None),
    }
    Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
