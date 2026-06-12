from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from st_edge_pruning.metrics import cuda_sync, get_gpu_memory, is_correct, now
from st_edge_pruning.types import EvalSample, PruneConfig
from st_edge_pruning.video_io import load_video_frames


class FirstTokenTimer(StoppingCriteria):
    """Record time-to-first-token as a practical prefill proxy."""

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
    """Map config dtype string to torch dtype for video pixel tensors."""

    return torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16


def _build_prompt(sample: EvalSample, frame_time: str, video_time: float, model, config: Dict[str, Any]) -> str:
    """Build the exact prompt sent to LLaVA."""

    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

    question = sample.question
    if config.get("add_time_instruction", True):
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. Please answer the following question related to this video."
        )
        question = f"{time_instruction}\n{question}"
    if getattr(model.config, "mm_use_im_start_end", False):
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    return DEFAULT_IMAGE_TOKEN + "\n" + question


def _prepare_inputs(tokenizer, prompt: str, device: torch.device):
    """Tokenize prompt while preserving the LLaVA image placeholder."""

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    text_tokens = int((input_ids != IMAGE_TOKEN_INDEX).sum().item())
    return input_ids, attention_mask, text_tokens


def run_one_sample(sample: EvalSample, tokenizer, model, image_processor, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one sample and return prediction, token stats, timing, and frames."""

    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria

    frames, frame_time, video_time = load_video_frames(
        sample.video_path,
        num_frames=int(config["num_frames"]),
        frame_sampling=config["frame_sampling"],
        video_fps=float(config["video_fps"]),
        force_sample=bool(config["force_sample"]),
    )
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video = video.to(device=_model_device(model), dtype=_model_dtype(config))
    video = [video]

    conv = conv_templates[config["conv_mode"]].copy()
    conv.append_message(conv.roles[0], _build_prompt(sample, frame_time, video_time, model, config))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids, attention_mask, text_tokens = _prepare_inputs(tokenizer, prompt, _model_device(model))

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keyword_stopper = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    prune_context = {
        "enabled": True,
        "config": config,
        "prune_config": PruneConfig.from_mapping(config),
        "raw_frames": frames,
        "sample_id": sample.sample_id,
        "results": {},
    }
    model.st_edge_pruning_context = prune_context

    if config.get("measure_gpu_memory", True) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cuda_sync(bool(config.get("torch_cuda_sync", True)))
    start = now()
    first_token_timer = FirstTokenTimer(start)
    stopping = StoppingCriteriaList([keyword_stopper, first_token_timer])

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=video,
            attention_mask=attention_mask,
            modalities="video",
            do_sample=bool(config["do_sample"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            num_beams=int(config["num_beams"]),
            max_new_tokens=int(config["max_new_tokens"]),
            use_cache=True,
            stopping_criteria=stopping,
        )
    cuda_sync(bool(config.get("torch_cuda_sync", True)))
    total_time = now() - start
    prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    result = prune_context.get("results", {}).get(0, {})
    token_stats = dict(result.get("stats", {}))
    token_stats.update(
        {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "text_tokens": text_tokens,
            "total_input_tokens_before": text_tokens + int(token_stats.get("llm_visual_tokens_before", 0)),
            "total_input_tokens_after": text_tokens + int(token_stats.get("llm_visual_tokens_after", 0)),
        }
    )
    timings = {
        "sample_id": sample.sample_id,
        "total_inference_time_sec": total_time,
        # This is time-to-first-generated-token. It is logged under the prefill
        # field so runs have the required column, and can be refined later.
        "llm_prefill_time_sec": first_token_timer.first_token_time,
        **get_gpu_memory(),
    }
    pred_record = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "task": sample.task,
        "video_path": sample.video_path,
        "question": sample.question,
        "choices": sample.choices,
        "answer": sample.answer,
        "prediction": prediction,
        "correct": is_correct(prediction, sample.answer, sample.choices),
        "method": config["method"],
        "target_keep_ratio": float(config["keep_ratio"]),
        "actual_keep_ratio": token_stats.get("actual_keep_ratio"),
        "window_size": int(config["window_size"]),
    }

    # Clear the context to avoid accidentally reusing frames on the next sample.
    model.st_edge_pruning_context = None
    return {
        "prediction": pred_record,
        "token_stats": token_stats,
        "timings": timings,
        "keep_probs": result.get("keep_probs"),
        "frames": frames,
    }
