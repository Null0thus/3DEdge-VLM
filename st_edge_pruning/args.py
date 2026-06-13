from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .dataset_configs import load_config_file, load_dataset_config, merge_configs


def _str_to_bool(value: str | bool | None) -> bool | None:
    """argparse helper for explicit true/false command-line values."""

    if value is None or isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _add_common_arguments(parser: argparse.ArgumentParser, default_none: bool) -> None:
    """Register every experiment hyperparameter in one place."""

    d = None if default_none else argparse.SUPPRESS
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--model-family", dest="model_family", choices=["llava_video", "videollama3"], default=d)
    parser.add_argument("--model-path", dest="model_path", default=d)
    parser.add_argument("--llava-root", dest="llava_root", default=d)
    parser.add_argument("--dataset", default=d)
    parser.add_argument("--method", choices=["full", "random", "ours", "evs", "difffp"], default=d)
    parser.add_argument("--output-dir", dest="output_dir", default=d)
    parser.add_argument("--experiment-name", dest="experiment_name", default=d)
    parser.add_argument("--run-name", dest="run_name", default=d)
    parser.add_argument("--seed", type=int, default=d)
    parser.add_argument("--device", default=d)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default=d)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=d)
    parser.add_argument("--limit-samples", dest="limit_samples", type=int, default=d)
    parser.add_argument("--num-chunks", dest="num_chunks", type=int, default=d)
    parser.add_argument("--chunk-idx", dest="chunk_idx", type=int, default=d)
    parser.add_argument("--chunk-strategy", dest="chunk_strategy", choices=["contiguous", "round_robin"], default=d)
    parser.add_argument("--missing-video-policy", dest="missing_video_policy", choices=["error", "skip"], default=d)

    parser.add_argument("--num-frames", dest="num_frames", type=int, default=d)
    parser.add_argument("--frame-sampling", dest="frame_sampling", choices=["uniform", "fps"], default=d)
    parser.add_argument("--video-fps", dest="video_fps", type=float, default=d)
    parser.add_argument("--frame-dir-fps", dest="frame_dir_fps", type=float, default=d)
    parser.add_argument("--force-sample", dest="force_sample", type=_str_to_bool, default=d)
    parser.add_argument("--add-time-instruction", dest="add_time_instruction", type=_str_to_bool, default=d)
    parser.add_argument("--conv-mode", dest="conv_mode", default=d)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=d)
    parser.add_argument("--temperature", type=float, default=d)
    parser.add_argument("--top-p", dest="top_p", type=float, default=d)
    parser.add_argument("--num-beams", dest="num_beams", type=int, default=d)
    parser.add_argument("--do-sample", dest="do_sample", type=_str_to_bool, default=d)
    parser.add_argument("--load-8bit", dest="load_8bit", type=_str_to_bool, default=d)
    parser.add_argument("--load-4bit", dest="load_4bit", type=_str_to_bool, default=d)
    parser.add_argument("--attn-implementation", dest="attn_implementation", default=d)

    parser.add_argument("--videollama3-backend", dest="videollama3_backend", choices=["hf_local"], default=d)
    parser.add_argument("--videollama3-max-visual-tokens", dest="videollama3_max_visual_tokens", type=int, default=d)
    parser.add_argument("--videollama3-difffp-threshold", dest="videollama3_difffp_threshold", type=float, default=d)
    parser.add_argument("--videollama3-difffp-min-tokens", dest="videollama3_difffp_min_tokens", type=int, default=d)
    parser.add_argument("--videollama3-prompt-style", dest="videollama3_prompt_style", choices=["official_mvbench", "dataset"], default=d)
    parser.add_argument("--videollama3-add-system-prompt", dest="videollama3_add_system_prompt", type=_str_to_bool, default=d)

    parser.add_argument("--mm-spatial-pool-stride", dest="mm_spatial_pool_stride", type=int, default=d)
    parser.add_argument("--mm-spatial-pool-mode", dest="mm_spatial_pool_mode", choices=["average", "max", "bilinear"], default=d)
    parser.add_argument("--mm-newline-position", dest="mm_newline_position", choices=["grid", "frame", "one_token", "no_token"], default=d)
    parser.add_argument("--mm-patch-merge-type", dest="mm_patch_merge_type", default=d)

    parser.add_argument("--keep-ratio", dest="keep_ratio", type=float, default=d)
    parser.add_argument("--window-size", dest="window_size", type=int, default=d)
    parser.add_argument("--sampling-mode", dest="sampling_mode", choices=["bernoulli", "topk"], default=d)
    parser.add_argument("--position-encoding", dest="position_encoding", choices=["sequential", "preserve"], default=d)
    parser.add_argument("--topk-rounding", dest="topk_rounding", choices=["floor", "round", "ceil"], default=d)
    parser.add_argument("--min-keep-tokens", dest="min_keep_tokens", type=int, default=d)
    parser.add_argument("--keep-at-least-one-per-frame", dest="keep_at_least_one_per_frame", type=_str_to_bool, default=d)

    parser.add_argument("--work-height", dest="work_height", type=int, default=d)
    parser.add_argument("--work-width", dest="work_width", type=int, default=d)
    parser.add_argument("--spatial-sigma", dest="spatial_sigma", type=float, default=d)
    parser.add_argument("--spatial-kernel-size", dest="spatial_kernel_size", type=int, default=d)
    parser.add_argument("--spatial-grad-kernel", dest="spatial_grad_kernel", choices=["sobel", "scharr"], default=d)
    parser.add_argument("--temporal-diff", dest="temporal_diff", choices=["center"], default=d)
    parser.add_argument("--alpha-s", dest="alpha_s", type=float, default=d)
    parser.add_argument("--alpha-t", dest="alpha_t", type=float, default=d)
    parser.add_argument("--min-dyn-cc-size", dest="min_dyn_cc_size", type=int, default=d)
    parser.add_argument("--min-sta-cc-size", dest="min_sta_cc_size", type=int, default=d)
    parser.add_argument("--beta-dyn", dest="beta_dyn", type=float, default=d)
    parser.add_argument("--beta-sta", dest="beta_sta", type=float, default=d)
    parser.add_argument("--pi-min", dest="pi_min", type=float, default=d)
    parser.add_argument("--pi-max", dest="pi_max", type=float, default=d)
    parser.add_argument("--lambda-solver-iters", dest="lambda_solver_iters", type=int, default=d)
    parser.add_argument("--lambda-solver-tol", dest="lambda_solver_tol", type=float, default=d)
    parser.add_argument("--importance-clip", dest="importance_clip", type=float, default=d)
    parser.add_argument("--evs-space", dest="evs_space", choices=["embedding"], default=d)
    parser.add_argument("--evs-metric", dest="evs_metric", choices=["cosine", "l2"], default=d)
    parser.add_argument("--evs-anchor-first-frame", dest="evs_anchor_first_frame", type=_str_to_bool, default=d)

    parser.add_argument("--save-heatmap", dest="save_heatmap", type=_str_to_bool, default=d)
    parser.add_argument("--heatmap-save-count", dest="heatmap_save_count", type=int, default=d)
    parser.add_argument("--save-prob-npy", dest="save_prob_npy", type=_str_to_bool, default=d)
    parser.add_argument("--save-prob-png", dest="save_prob_png", type=_str_to_bool, default=d)
    parser.add_argument("--save-overlay", dest="save_overlay", type=_str_to_bool, default=d)
    parser.add_argument("--overlay-alpha", dest="overlay_alpha", type=float, default=d)
    parser.add_argument("--heatmap-colormap", dest="heatmap_colormap", default=d)

    parser.add_argument("--measure-prefill-time", dest="measure_prefill_time", type=_str_to_bool, default=d)
    parser.add_argument("--measure-total-time", dest="measure_total_time", type=_str_to_bool, default=d)
    parser.add_argument("--measure-gpu-memory", dest="measure_gpu_memory", type=_str_to_bool, default=d)
    parser.add_argument("--torch-cuda-sync", dest="torch_cuda_sync", type=_str_to_bool, default=d)


def _read_model_type(model_path: str) -> str | None:
    """Read model_type from a local HuggingFace config when available."""

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        return str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model config JSON: {config_path}") from exc


def _validate_eval_config(config: Dict[str, Any]) -> None:
    """Fail early for model/method/path combinations that would be ambiguous."""

    model_family = str(config.get("model_family", "llava_video"))
    method = str(config.get("method", "full"))
    model_path = str(config.get("model_path", ""))
    if bool(config.get("load_8bit", False)) and bool(config.get("load_4bit", False)):
        raise ValueError("Choose at most one of --load-8bit and --load-4bit")

    if method == "difffp" and model_family != "videollama3":
        raise ValueError("--method difffp is only valid with --model-family videollama3")
    if method == "evs" and model_family == "videollama3":
        raise ValueError("--method evs is not implemented for --model-family videollama3")

    if model_family == "videollama3":
        if str(config.get("videollama3_backend", "hf_local")) == "hf_local" and not Path(model_path).exists():
            raise ValueError(f"VideoLLaMA3 hf_local backend requires a local model path, got: {model_path}")
        model_type = _read_model_type(model_path)
        if model_type is not None and "videollama3" not in model_type.lower():
            raise ValueError(f"Expected a VideoLLaMA3 model config, got model_type={model_type!r} from {model_path}")
        allowed = {"full", "difffp", "random", "ours"}
        if method not in allowed:
            raise ValueError(f"VideoLLaMA3 method must be one of {sorted(allowed)}, got {method!r}")
    elif model_family == "llava_video":
        model_type = _read_model_type(model_path)
        if model_type is not None and "videollama3" in model_type.lower():
            raise ValueError("model_path looks like VideoLLaMA3; pass --model-family videollama3")
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")


def parse_eval_config(argv: list[str] | None = None) -> Dict[str, Any]:
    """Parse CLI and merge it with default and dataset configs."""

    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", default="configs/default_eval.yaml")
    early.add_argument("--dataset", default=None)
    early.add_argument("--dataset-config", default=None)
    early_args, _ = early.parse_known_args(argv)

    default_cfg = load_config_file(early_args.config)
    dataset_name = early_args.dataset or default_cfg.get("dataset", "mvbench")
    dataset_cfg = load_dataset_config(dataset_name, early_args.dataset_config)

    parser = argparse.ArgumentParser()
    _add_common_arguments(parser, default_none=True)
    cli_args = vars(parser.parse_args(argv))
    cli_cfg = {k: v for k, v in cli_args.items() if v is not None}
    cli_cfg.pop("config", None)
    cli_cfg.pop("dataset_config", None)

    cfg = merge_configs(default_cfg, dataset_cfg, cli_cfg)
    cfg["dataset_config_path"] = early_args.dataset_config or str(Path("configs") / "datasets" / f"{cfg['dataset'].lower()}.yaml")
    cfg["default_config_path"] = early_args.config
    _validate_eval_config(cfg)
    return cfg


def save_args_json(config: Dict[str, Any], path: str | Path) -> None:
    """Persist the final resolved configuration for reproducibility."""

    Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
