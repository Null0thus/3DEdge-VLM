from __future__ import annotations

from st_edge_pruning.args import parse_eval_config, save_args_json
from st_edge_pruning.datasets import build_dataset
from st_edge_pruning.io import build_run_paths
from st_edge_pruning.llava_adapter.inference import run_one_sample
from st_edge_pruning.llava_adapter.model_loader import load_llava_video_model
from st_edge_pruning.visualization import save_probability_heatmaps


def main() -> None:
    """Run one sample and print token/shape diagnostics."""

    config = parse_eval_config()
    config["limit_samples"] = 1
    config["save_heatmap"] = True
    config["heatmap_save_count"] = 1
    run_paths = build_run_paths(config)
    save_args_json(config, run_paths.args_file)

    tokenizer, model, image_processor, _ = load_llava_video_model(config)
    model.eval()
    sample = build_dataset(config)[0]
    result = run_one_sample(sample, tokenizer, model, image_processor, config)
    save_probability_heatmaps(result["keep_probs"], result["frames"], sample.sample_id, run_paths.output_dir, config)

    print("sample_id:", sample.sample_id)
    print("prediction:", result["prediction"]["prediction"])
    print("correct:", result["prediction"]["correct"])
    print("token_stats:", result["token_stats"])
    print("timings:", result["timings"])
    print("output_dir:", run_paths.output_dir)


if __name__ == "__main__":
    main()
