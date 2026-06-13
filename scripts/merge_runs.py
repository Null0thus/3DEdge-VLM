from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, TextIO


def _str_to_bool(value: str | bool) -> bool:
    """Parse shell-friendly true/false values."""

    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    """Parse merge arguments for multi-GPU chunk outputs."""

    parser = argparse.ArgumentParser(description="Merge chunked evaluation outputs.")
    parser.add_argument("--output-dir", required=True, help="Merged output directory.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Chunk output directories to merge in order.")
    parser.add_argument("--copy-heatmaps", type=_str_to_bool, default=True, help="Copy heatmaps into the merged directory.")
    parser.add_argument("--heatmap-source", default=None, help="Optional chunk directory whose heatmaps should be copied.")
    parser.add_argument("--deduplicate", type=_str_to_bool, default=True, help="Drop duplicate sample_id records if present.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read one JSONL file into a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    records: List[Dict[str, Any]] = []
    # utf-8-sig also accepts normal UTF-8 and is tolerant of BOM files created
    # by some Windows tools.
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_optional_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read an optional JSONL file, returning an empty list when absent."""

    if not path.exists():
        return []
    return _read_jsonl(path)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write dictionaries as UTF-8 JSONL."""

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class _TeeStream:
    """Mirror merge output to both the terminal and merge.log."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name: str):
        """Forward terminal attributes such as isatty() and encoding."""

        return getattr(self.terminal, name)


@contextmanager
def _tee_output(log_path: Path) -> Iterator[None]:
    """Capture stdout and stderr in the merged experiment directory."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(_TeeStream(sys.stdout, log_file)), redirect_stderr(_TeeStream(sys.stderr, log_file)):
            yield


def _deduplicate_by_sample_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the first record for each sample_id."""

    seen = set()
    unique = []
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id is None or sample_id not in seen:
            unique.append(record)
            if sample_id is not None:
                seen.add(sample_id)
    return unique


def _avg(records: List[Dict[str, Any]], key: str):
    """Average numeric fields while ignoring missing values."""

    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Any, digits: int = 4) -> str:
    """Format optional numeric values for summary.txt."""

    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary_text(summary: Dict[str, Any]) -> str:
    """Create a compact text summary for direct inspection."""

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
    ]
    return "\n".join(lines) + "\n"


def _write_summary(
    path: Path,
    predictions: List[Dict[str, Any]],
    token_stats: List[Dict[str, Any]],
    timings: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute global summary from sample-level merged records."""

    total = len(predictions)
    correct = sum(1 for record in predictions if record.get("correct"))
    summary = {
        "num_samples": total,
        "num_evaluated": total,
        "num_correct": correct,
        "num_skipped": len(skipped),
        "num_total_records": total + len(skipped),
        "accuracy": correct / max(total, 1),
        "avg_actual_keep_ratio": _avg(token_stats, "actual_keep_ratio"),
        "avg_ordinary_video_tokens_before": _avg(token_stats, "ordinary_video_tokens_before"),
        "avg_ordinary_video_tokens_after": _avg(token_stats, "ordinary_video_tokens_after"),
        "avg_llm_visual_tokens_before": _avg(token_stats, "llm_visual_tokens_before"),
        "avg_llm_visual_tokens_after": _avg(token_stats, "llm_visual_tokens_after"),
        "avg_llm_visual_keep_ratio": _avg(token_stats, "llm_visual_keep_ratio"),
        "avg_num_sampled_frames": _avg(token_stats, "num_sampled_frames"),
        "avg_total_inference_time_sec": _avg(timings, "total_inference_time_sec"),
        "avg_llm_prefill_time_sec": _avg(timings, "llm_prefill_time_sec"),
        "max_gpu_memory_peak_mib": max([record.get("gpu_memory_peak_mib") or 0 for record in timings], default=None),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".txt").write_text(_summary_text(summary), encoding="utf-8")
    return summary


def _load_args(run_dir: Path) -> Dict[str, Any]:
    """Load args.json from one chunk directory when available."""

    args_file = run_dir / "args.json"
    if not args_file.exists():
        return {}
    # utf-8-sig is tolerant of BOM files produced by some shell tools.
    return json.loads(args_file.read_text(encoding="utf-8-sig"))


def _copy_heatmaps(run_dirs: List[Path], output_dir: Path, heatmap_source: str | None) -> None:
    """Copy heatmaps from one chunk, usually chunk0 with the first 20 samples."""

    if heatmap_source:
        source = Path(heatmap_source) / "heatmaps"
    else:
        source = next((run_dir / "heatmaps" for run_dir in run_dirs if (run_dir / "heatmaps").exists()), None)
    if source is None or not source.exists():
        return
    target = output_dir / "heatmaps"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _copy_chunk_logs(run_dirs: List[Path], output_dir: Path) -> None:
    """Collect each chunk run.log under the merged logs/chunks directory."""

    target = output_dir / "logs" / "chunks"
    target.mkdir(parents=True, exist_ok=True)
    for index, run_dir in enumerate(run_dirs):
        source = run_dir / "logs" / "run.log"
        if source.exists():
            shutil.copy2(source, target / f"chunk{index}_run.log")


def main() -> None:
    """Merge chunk outputs into one run directory."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with _tee_output(output_dir / "logs" / "merge.log"):
        run_dirs = [Path(run_dir) for run_dir in args.run_dirs]
        print(f"Logging to: {output_dir / 'logs' / 'merge.log'}")

        predictions: List[Dict[str, Any]] = []
        token_stats: List[Dict[str, Any]] = []
        timings: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        source_args = []

        for run_dir in run_dirs:
            predictions.extend(_read_jsonl(run_dir / "predictions.jsonl"))
            token_stats.extend(_read_jsonl(run_dir / "token_stats.jsonl"))
            timings.extend(_read_jsonl(run_dir / "timings.jsonl"))
            skipped.extend(_read_optional_jsonl(run_dir / "skipped.jsonl"))
            source_args.append({"run_dir": str(run_dir), "args": _load_args(run_dir)})

        if args.deduplicate:
            predictions = _deduplicate_by_sample_id(predictions)
            token_stats = _deduplicate_by_sample_id(token_stats)
            timings = _deduplicate_by_sample_id(timings)
            skipped = _deduplicate_by_sample_id(skipped)

        _write_jsonl(output_dir / "predictions.jsonl", predictions)
        _write_jsonl(output_dir / "token_stats.jsonl", token_stats)
        _write_jsonl(output_dir / "timings.jsonl", timings)
        _write_jsonl(output_dir / "skipped.jsonl", skipped)
        summary = _write_summary(output_dir / "summary.json", predictions, token_stats, timings, skipped)

        merged_args = {
            "merged_from": [str(run_dir) for run_dir in run_dirs],
            "source_args": source_args,
            "num_merged_predictions": len(predictions),
            "num_merged_skipped": len(skipped),
            "deduplicate": bool(args.deduplicate),
        }
        (output_dir / "args.json").write_text(json.dumps(merged_args, indent=2, ensure_ascii=False), encoding="utf-8")

        _copy_chunk_logs(run_dirs, output_dir)
        if args.copy_heatmaps:
            _copy_heatmaps(run_dirs, output_dir, args.heatmap_source)

        print(f"Merged output saved to: {output_dir}")
        print(f"Summary text: {output_dir / 'summary.txt'}")
        print(f"Samples: {summary['num_samples']}")
        print(f"Correct: {summary['num_correct']} / {summary['num_evaluated']}")
        print(f"Skipped samples: {summary['num_skipped']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
