from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from st_edge_pruning.types import RunPaths


def build_run_paths(config: Dict[str, Any]) -> RunPaths:
    """Create the output directory layout for one experiment run."""

    run_name = config.get("run_name")
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{stamp}_{config['dataset']}_{config['method']}_keep{float(config['keep_ratio']):.2f}".replace(".", "")
    # experiment_name groups related chunks/merged outputs under one folder,
    # e.g. outputs/mvbench_ours_keep025/chunk0 instead of many loose folders.
    base_dir = Path(config["output_dir"])
    if config.get("experiment_name"):
        base_dir = base_dir / str(config["experiment_name"])
    output_dir = base_dir / run_name
    heatmap_dir = output_dir / "heatmaps"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        output_dir=output_dir,
        predictions_file=output_dir / "predictions.jsonl",
        token_stats_file=output_dir / "token_stats.jsonl",
        timings_file=output_dir / "timings.jsonl",
        skipped_file=output_dir / "skipped.jsonl",
        summary_file=output_dir / "summary.json",
        summary_text_file=output_dir / "summary.txt",
        args_file=output_dir / "args.json",
        heatmap_dir=heatmap_dir,
        logs_dir=logs_dir,
        run_log_file=logs_dir / "run.log",
    )


class JsonlWriter:
    """Line-oriented JSON writer for per-sample records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = self.path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
