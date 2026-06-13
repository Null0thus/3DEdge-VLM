from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `python scripts/run_eval.py` from the project root on Linux servers
    # without requiring the user to export PYTHONPATH manually.
    sys.path.insert(0, str(PROJECT_ROOT))

from st_edge_pruning.args import parse_eval_config, save_args_json
from st_edge_pruning.datasets import build_dataset
from st_edge_pruning.io import JsonlWriter, build_run_paths, tee_output, write_summary
from st_edge_pruning.visualization import save_probability_heatmaps


def _load_model_and_runner(config: Dict[str, Any]):
    """Select the model adapter without importing unused third-party code."""

    model_family = str(config.get("model_family", "llava_video"))
    if model_family == "llava_video":
        from st_edge_pruning.llava_adapter.inference import run_one_sample
        from st_edge_pruning.llava_adapter.model_loader import load_llava_video_model

        tokenizer, model, processor, context_len = load_llava_video_model(config)
        return tokenizer, model, processor, context_len, run_one_sample
    if model_family == "videollama3":
        from st_edge_pruning.videollama3_adapter.inference import run_one_sample
        from st_edge_pruning.videollama3_adapter.model_loader import load_videollama3_model

        tokenizer, model, processor, context_len = load_videollama3_model(config)
        return tokenizer, model, processor, context_len, run_one_sample
    raise ValueError(f"Unsupported model_family: {model_family}")


def _chunk_samples(samples, num_chunks: int, chunk_idx: int, chunk_strategy: str):
    """Split samples for resumable or multi-GPU evaluation."""

    if num_chunks <= 1:
        return samples
    if chunk_strategy == "round_robin":
        # Round-robin mixing avoids slow chunks when neighboring MVBench samples
        # belong to harder tasks or longer videos.
        return samples[chunk_idx::num_chunks]
    if chunk_strategy == "contiguous":
        chunk_size = math.ceil(len(samples) / num_chunks)
        start = chunk_idx * chunk_size
        return samples[start : start + chunk_size]
    raise ValueError(f"Unsupported chunk_strategy: {chunk_strategy}")


def _is_data_read_error(exc: Exception) -> bool:
    """Detect dataset media errors without hiding model or CUDA failures."""

    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, ZeroDivisionError) and "division by zero" in message:
        return True
    if isinstance(exc, RuntimeError) and "Error reading" in message:
        return True
    if "No image frames found" in message or "empty video or frame directory" in message:
        return True
    return False


def _skipped_record(sample, reason: str, exc: Exception | None = None) -> Dict[str, Any]:
    """Build one skipped-sample log row with enough path diagnostics."""

    record: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "task": sample.task,
        "video_path": sample.video_path,
        "reason": reason,
        "metadata": sample.metadata,
    }
    if exc is not None:
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    return record


def main() -> None:
    """Run one configured evaluation job on the selected model family."""

    config = parse_eval_config()
    run_paths = build_run_paths(config)
    with tee_output(run_paths.run_log_file):
        print(f"Logging to: {run_paths.run_log_file}")
        save_args_json(config, run_paths.args_file)

        tokenizer, model, image_processor, _, run_one_sample = _load_model_and_runner(config)
        model.eval()

        samples = build_dataset(config)
        samples = _chunk_samples(samples, int(config["num_chunks"]), int(config["chunk_idx"]), str(config.get("chunk_strategy", "round_robin")))
        if config.get("limit_samples") is not None:
            samples = samples[: int(config["limit_samples"])]

        predictions = []
        token_stats = []
        timings = []
        skipped = []
        heatmap_budget = int(config.get("heatmap_save_count", 0))
        saved_heatmaps = 0
        missing_video_policy = str(config.get("missing_video_policy", "skip"))

        with (
            JsonlWriter(run_paths.predictions_file) as pred_writer,
            JsonlWriter(run_paths.token_stats_file) as token_writer,
            JsonlWriter(run_paths.timings_file) as timing_writer,
            JsonlWriter(run_paths.skipped_file) as skipped_writer,
        ):
            for sample in tqdm(samples, desc="Evaluating"):
                if missing_video_policy == "skip" and sample.metadata.get("video_missing"):
                    # Missing assets are logged and excluded from accuracy, instead
                    # of crashing every worker on the same incomplete dataset shard.
                    skip_record = _skipped_record(sample, "missing_media_path")
                    skipped.append(skip_record)
                    skipped_writer.write(skip_record)
                    continue

                try:
                    result = run_one_sample(sample, tokenizer, model, image_processor, config)
                except Exception as exc:
                    if missing_video_policy == "skip" and _is_data_read_error(exc):
                        model.st_edge_pruning_context = None
                        skip_record = _skipped_record(sample, "media_read_error", exc)
                        skipped.append(skip_record)
                        skipped_writer.write(skip_record)
                        continue
                    raise

                predictions.append(result["prediction"])
                token_stats.append(result["token_stats"])
                timings.append(result["timings"])
                pred_writer.write(result["prediction"])
                token_writer.write(result["token_stats"])
                timing_writer.write(result["timings"])

                if config.get("save_heatmap", False) and saved_heatmaps < heatmap_budget:
                    save_probability_heatmaps(result["keep_probs"], result["frames"], sample.sample_id, run_paths.output_dir, config)
                    saved_heatmaps += 1

        summary = write_summary(run_paths.summary_file, predictions, token_stats, timings, skipped)
        print(f"Results saved to: {run_paths.output_dir}")
        print(f"Summary text: {run_paths.summary_text_file}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"Correct: {summary['num_correct']} / {summary['num_evaluated']}")
        print(f"Skipped samples: {summary['num_skipped']}")


if __name__ == "__main__":
    main()
