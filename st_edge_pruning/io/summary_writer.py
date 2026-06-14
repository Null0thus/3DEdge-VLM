from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def _avg(records: List[Dict[str, Any]], key: str):
    values = [r[key] for r in records if r.get(key) is not None]
    return mean(values) if values else None


def _fmt(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for the human-readable summary."""

    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _accuracy_by_key(predictions: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Compute grouped accuracy for dataset-specific fields."""

    groups: Dict[str, Dict[str, Any]] = {}
    for record in predictions:
        value = record.get(key)
        if value is None:
            continue
        name = str(value)
        group = groups.setdefault(name, {"num_samples": 0, "num_correct": 0})
        group["num_samples"] += 1
        group["num_correct"] += int(bool(record.get("correct")))
    for group in groups.values():
        group["accuracy"] = group["num_correct"] / max(group["num_samples"], 1)
    return groups


def _summary_text(summary: Dict[str, Any]) -> str:
    """Render the key metrics as a compact text report."""

    accuracy = summary.get("accuracy")
    accuracy_pct = None if accuracy is None else accuracy * 100.0
    lines = [
        "Experiment Summary",
        "==================",
        f"Evaluated samples: {summary.get('num_evaluated', 0)}",
        f"Skipped samples: {summary.get('num_skipped', 0)}",
        f"Total records: {summary.get('num_total_records', 0)}",
        f"Correct: {summary.get('num_correct', 0)} / {summary.get('num_evaluated', 0)}",
        f"Accuracy: {_fmt(accuracy, 6)} ({_fmt(accuracy_pct, 2)}%)",
    ]
    by_format = summary.get("accuracy_by_question_format") or {}
    if by_format:
        lines.extend(["", "Accuracy By Question Format", "---------------------------"])
        for name, metrics in sorted(by_format.items()):
            fmt_acc = metrics.get("accuracy")
            fmt_pct = None if fmt_acc is None else fmt_acc * 100.0
            lines.append(
                f"{name}: {_fmt(fmt_acc, 6)} ({_fmt(fmt_pct, 2)}%), "
                f"{metrics.get('num_correct', 0)} / {metrics.get('num_samples', 0)}"
            )
    lines.extend([
        "",
        "Token Statistics",
        "----------------",
        f"Average actual keep ratio: {_fmt(summary.get('avg_actual_keep_ratio'), 6)}",
        f"Average ordinary video tokens before: {_fmt(summary.get('avg_ordinary_video_tokens_before'), 2)}",
        f"Average ordinary video tokens after: {_fmt(summary.get('avg_ordinary_video_tokens_after'), 2)}",
        f"Average LLM visual tokens before: {_fmt(summary.get('avg_llm_visual_tokens_before'), 2)}",
        f"Average LLM visual tokens after: {_fmt(summary.get('avg_llm_visual_tokens_after'), 2)}",
        f"Average LLM visual keep ratio: {_fmt(summary.get('avg_llm_visual_keep_ratio'), 6)}",
        f"Average sampled frames: {_fmt(summary.get('avg_num_sampled_frames'), 2)}",
        "",
        "Runtime",
        "-------",
        f"Average total inference time sec: {_fmt(summary.get('avg_total_inference_time_sec'), 4)}",
        f"Average LLM prefill time sec: {_fmt(summary.get('avg_llm_prefill_time_sec'), 4)}",
        f"Max GPU memory peak MiB: {_fmt(summary.get('max_gpu_memory_peak_mib'), 2)}",
    ])
    return "\n".join(lines) + "\n"


def write_summary(
    path: str | Path,
    predictions: List[Dict[str, Any]],
    token_stats: List[Dict[str, Any]],
    timings: List[Dict[str, Any]],
    skipped: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write aggregate metrics for one run."""

    skipped = skipped or []
    total = len(predictions)
    correct = sum(1 for r in predictions if r.get("correct"))
    summary = {
        "num_samples": total,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": len(skipped),
        "num_total_records": total + len(skipped),
        "accuracy": correct / max(total, 1),
        "accuracy_by_question_format": _accuracy_by_key(predictions, "question_format"),
        "accuracy_by_task": _accuracy_by_key(predictions, "task"),
        "avg_actual_keep_ratio": _avg(token_stats, "actual_keep_ratio"),
        "avg_ordinary_video_tokens_before": _avg(token_stats, "ordinary_video_tokens_before"),
        "avg_ordinary_video_tokens_after": _avg(token_stats, "ordinary_video_tokens_after"),
        "avg_llm_visual_tokens_before": _avg(token_stats, "llm_visual_tokens_before"),
        "avg_llm_visual_tokens_after": _avg(token_stats, "llm_visual_tokens_after"),
        "avg_llm_visual_keep_ratio": _avg(token_stats, "llm_visual_keep_ratio"),
        "avg_num_sampled_frames": _avg(token_stats, "num_sampled_frames"),
        "avg_total_inference_time_sec": _avg(timings, "total_inference_time_sec"),
        "avg_llm_prefill_time_sec": _avg(timings, "llm_prefill_time_sec"),
        "max_gpu_memory_peak_mib": max([r.get("gpu_memory_peak_mib") or 0 for r in timings], default=None),
    }
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".txt").write_text(_summary_text(summary), encoding="utf-8")
    return summary
