from __future__ import annotations

import math
from pathlib import Path

from tqdm import tqdm

from st_edge_pruning.args import parse_eval_config, save_args_json
from st_edge_pruning.datasets import build_dataset
from st_edge_pruning.io import JsonlWriter, build_run_paths, write_summary
from st_edge_pruning.llava_adapter.inference import run_one_sample
from st_edge_pruning.llava_adapter.model_loader import load_llava_video_model
from st_edge_pruning.visualization import save_probability_heatmaps


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


def main() -> None:
    """Run Full/Random/Ours evaluation on the selected dataset."""

    config = parse_eval_config()
    run_paths = build_run_paths(config)
    save_args_json(config, run_paths.args_file)

    tokenizer, model, image_processor, _ = load_llava_video_model(config)
    model.eval()

    samples = build_dataset(config)
    samples = _chunk_samples(samples, int(config["num_chunks"]), int(config["chunk_idx"]), str(config.get("chunk_strategy", "round_robin")))
    if config.get("limit_samples") is not None:
        samples = samples[: int(config["limit_samples"])]

    predictions = []
    token_stats = []
    timings = []
    heatmap_budget = int(config.get("heatmap_save_count", 0))

    with JsonlWriter(run_paths.predictions_file) as pred_writer, JsonlWriter(run_paths.token_stats_file) as token_writer, JsonlWriter(run_paths.timings_file) as timing_writer:
        for sample_index, sample in enumerate(tqdm(samples, desc="Evaluating")):
            result = run_one_sample(sample, tokenizer, model, image_processor, config)
            predictions.append(result["prediction"])
            token_stats.append(result["token_stats"])
            timings.append(result["timings"])
            pred_writer.write(result["prediction"])
            token_writer.write(result["token_stats"])
            timing_writer.write(result["timings"])

            if config.get("save_heatmap", False) and sample_index < heatmap_budget:
                save_probability_heatmaps(result["keep_probs"], result["frames"], sample.sample_id, run_paths.output_dir, config)

    summary = write_summary(run_paths.summary_file, predictions, token_stats, timings)
    print(f"Results saved to: {run_paths.output_dir}")
    print(f"Accuracy: {summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
