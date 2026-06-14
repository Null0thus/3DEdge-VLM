from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from st_edge_pruning.generation import build_generation_kwargs
from st_edge_pruning.metrics import cuda_sync, get_gpu_memory, is_correct, now
from st_edge_pruning.types import EvalSample, PruneConfig
from st_edge_pruning.video_io import load_video_frames


class FirstTokenTimer(StoppingCriteria):
    """Record time-to-first-token as the prefill proxy used by our logs."""

    def __init__(self, start_time: float):
        self.start_time = start_time
        self.first_token_time = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.first_token_time is None:
            self.first_token_time = now() - self.start_time
        return False


def _model_device(model) -> torch.device:
    """Find the first parameter device for input tensors."""

    return next(model.parameters()).device


def _model_dtype(config: Dict[str, Any]) -> torch.dtype:
    """Map config dtype string to torch dtype for VideoLLaMA3 pixel tensors."""

    return torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16


def _metadata_float(sample: EvalSample, *keys: str) -> float | None:
    """Read optional numeric timing metadata from a dataset sample."""

    for key in keys:
        value = sample.metadata.get(key)
        if value is not None:
            return float(value)
    return None


def _to_hwc_uint8(frame) -> np.ndarray:
    """Convert VideoLLaMA3 processor frames to HWC uint8 for edge evidence."""

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _frames_to_hwc(frames: List[Any]) -> np.ndarray:
    """Stack sampled frames as [T,H,W,3]."""

    if not frames:
        raise ValueError("VideoLLaMA3 loaded zero frames")
    return np.stack([_to_hwc_uint8(frame) for frame in frames], axis=0)


def _frames_to_chw_list(frames: np.ndarray) -> List[np.ndarray]:
    """Convert our HWC fallback frames to VideoLLaMA3's CHW frame convention."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected fallback frames in [T,H,W,3], got shape={frames.shape}")
    return [np.transpose(frame, (2, 0, 1)) for frame in frames]


def _parse_frame_time_string(frame_time: str, num_frames: int) -> List[float]:
    """Parse timestamps returned by the shared decord/frame-directory loader."""

    timestamps = []
    for item in frame_time.split(","):
        item = item.strip()
        if item.endswith("s"):
            item = item[:-1]
        if item:
            timestamps.append(float(item))
    if len(timestamps) == num_frames:
        return timestamps
    # Keep the downstream processor alive even if a future frame_time format
    # changes; these timestamps are only prompt metadata for VideoLLaMA3.
    return [float(idx) for idx in range(num_frames)]


def _format_official_mvbench_prompt(sample: EvalSample) -> str:
    """Build the prompt style used by VideoLLaMA3's MVBench evaluator."""

    raw_question = str(sample.metadata.get("raw_question", sample.question))
    options = "\n".join(f"({chr(65 + idx)}) {choice}" for idx, choice in enumerate(sample.choices))
    return (
        f"Question: {raw_question}\n"
        f"Options:\n{options}\n"
        "Answer with the option's letter from the given choices directly and only give the best option."
    )


def _build_question(sample: EvalSample, config: Dict[str, Any]) -> str:
    """Select a dataset prompt or the official VideoLLaMA3 MVBench prompt."""

    if config.get("videollama3_prompt_style", "official_mvbench") == "official_mvbench" and sample.dataset == "mvbench":
        return _format_official_mvbench_prompt(sample)
    return sample.question


def _load_video_with_processor(sample: EvalSample, processor, config: Dict[str, Any]) -> Tuple[List[Any], List[float], np.ndarray, float]:
    """Use VideoLLaMA3's loader so model frames and edge frames stay aligned."""

    start_time = _metadata_float(sample, "start", "start_time")
    end_time = _metadata_float(sample, "end", "end_time")
    requested_fps = float(config["video_fps"]) if config.get("frame_sampling") == "fps" else None
    try:
        frames, timestamps = processor.load_video(
            sample.video_path,
            start_time=start_time,
            end_time=end_time,
            fps=requested_fps,
            max_frames=int(config["num_frames"]),
        )
        raw_frames = _frames_to_hwc(frames)
        video_time = float(max(timestamps) if timestamps else 0.0)
        return frames, [float(t) for t in timestamps], raw_frames, video_time
    except ZeroDivisionError:
        # VideoLLaMA3's load_video_from_ids can compute segment_len=0 when a
        # low-FPS video is sampled with a higher requested fps. Fall back to our
        # guarded loader, which clamps the sampling step to at least one frame.
        raw_frames, frame_time, video_time = load_video_frames(
            sample.video_path,
            num_frames=int(config["num_frames"]),
            frame_sampling=str(config.get("frame_sampling", "uniform")),
            video_fps=float(config.get("video_fps", 1.0)),
            force_sample=bool(config.get("force_sample", True)),
            start_time=start_time,
            end_time=end_time,
            frame_dir_fps=float(sample.metadata.get("frame_fps", config.get("frame_dir_fps", 3.0))),
        )
        frames = _frames_to_chw_list(raw_frames)
        timestamps = _parse_frame_time_string(frame_time, len(frames))
        return frames, timestamps, raw_frames, float(video_time)


def _build_conversation(sample: EvalSample, frames: List[Any], timestamps: List[float], config: Dict[str, Any]):
    """Create a VideoLLaMA3 conversation with preloaded video frames."""

    # The processor owns add_system_prompt, so the conversation only contains
    # user content. This matches the official VideoLLaMA3 inference example and
    # avoids accidentally adding two system prompts.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "num_frames": len(frames),
                    "timestamps": timestamps,
                },
                {"type": "text", "text": _build_question(sample, config)},
            ],
        }
    ]
    return messages


def _prepare_inputs(processor, conversation, device: torch.device, dtype: torch.dtype, config: Dict[str, Any]) -> Dict[str, Any]:
    """Tokenize conversation and move tensors to the target device."""

    inputs = processor(
        conversation=conversation,
        add_system_prompt=bool(config.get("videollama3_add_system_prompt", True)),
        add_generation_prompt=True,
        return_tensors="pt",
    )
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    if "pixel_values" in moved:
        moved["pixel_values"] = moved["pixel_values"].to(dtype=dtype)
    return moved


def run_one_sample(sample: EvalSample, tokenizer, model, processor, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one VideoLLaMA3 sample with selectable token compression."""

    frames, timestamps, raw_frames, video_time = _load_video_with_processor(sample, processor, config)
    conversation = _build_conversation(sample, frames, timestamps, config)
    inputs = _prepare_inputs(processor, conversation, _model_device(model), _model_dtype(config), config)

    image_token_id = getattr(processor, "image_token_id", tokenizer.convert_tokens_to_ids("<image>"))
    text_tokens = int((inputs["input_ids"] != image_token_id).sum().item())

    prune_context = {
        "enabled": True,
        "config": config,
        "prune_config": PruneConfig.from_mapping(config),
        "raw_frames": raw_frames,
        "sample_id": sample.sample_id,
        "results": {},
    }
    model.st_edge_pruning_context = prune_context

    if config.get("measure_gpu_memory", True) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cuda_sync(bool(config.get("torch_cuda_sync", True)))
    start = now()
    first_token_timer = FirstTokenTimer(start)
    stopping = StoppingCriteriaList([first_token_timer])

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **build_generation_kwargs(config),
                stopping_criteria=stopping,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception:
        model.st_edge_pruning_context = None
        raise

    cuda_sync(bool(config.get("torch_cuda_sync", True)))
    total_time = now() - start
    prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    result = prune_context.get("results", {}).get(0, {})
    token_stats = dict(result.get("stats", {}))
    token_stats.update(
        {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "text_tokens": text_tokens,
            "total_input_tokens_before": text_tokens + int(token_stats.get("llm_visual_tokens_before", 0)),
            "total_input_tokens_after": text_tokens + int(token_stats.get("llm_visual_tokens_after", 0)),
            "model_family": "videollama3",
            "videollama3_prompt_style": str(config.get("videollama3_prompt_style", "official_mvbench")),
            "video_time": video_time,
            "num_sampled_frames": len(frames),
        }
    )
    timings = {
        "sample_id": sample.sample_id,
        "total_inference_time_sec": total_time,
        "llm_prefill_time_sec": first_token_timer.first_token_time,
        **get_gpu_memory(),
    }
    pred_record = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "task": sample.task,
        "question_format": sample.metadata.get("question_format"),
        "video_path": sample.video_path,
        "question": _build_question(sample, config),
        "choices": sample.choices,
        "answer": sample.answer,
        "prediction": prediction,
        "correct": is_correct(prediction, sample.answer, sample.choices),
        "method": config["method"],
        "model_family": "videollama3",
        "target_keep_ratio": float(config["keep_ratio"]),
        "actual_keep_ratio": token_stats.get("actual_keep_ratio"),
        "window_size": int(config["window_size"]),
    }

    model.st_edge_pruning_context = None
    return {
        "prediction": pred_record,
        "token_stats": token_stats,
        "timings": timings,
        "keep_probs": result.get("keep_probs"),
        "frames": raw_frames,
    }
