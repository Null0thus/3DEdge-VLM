from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from st_edge_pruning.types import EvalSample

VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mkv", ".mov"}

# MVBench annotations store task-local paths. This mapping anchors every task
# to its official video root so relative paths such as "left/xxx.mp4" are not
# resolved by a fragile global filename search.
TASK_SOURCES: Dict[str, Dict[str, Any]] = {
    "action_sequence": {"prefix": "star/Charades_v1_480", "data_type": "video"},
    "action_prediction": {"prefix": "star/Charades_v1_480", "data_type": "video"},
    "action_antonym": {"prefix": "ssv2_video", "data_type": "video"},
    "fine_grained_action": {"prefix": "Moments_in_Time_Raw/videos", "data_type": "video"},
    "unexpected_action": {"prefix": "FunQA_test/test", "data_type": "video"},
    "object_existence": {"prefix": "clevrer/video_validation", "data_type": "video"},
    "object_interaction": {"prefix": "star/Charades_v1_480", "data_type": "video"},
    "object_shuffle": {"prefix": "perception/videos", "data_type": "video"},
    "moving_direction": {"prefix": "clevrer/video_validation", "data_type": "video"},
    "action_localization": {"prefix": "sta/sta_video", "data_type": "video"},
    "scene_transition": {"prefix": "scene_qa/video", "data_type": "video"},
    "action_count": {"prefix": "perception/videos", "data_type": "video"},
    "moving_count": {"prefix": "clevrer/video_validation", "data_type": "video"},
    "moving_attribute": {"prefix": "clevrer/video_validation", "data_type": "video"},
    "state_change": {"prefix": "perception/videos", "data_type": "video"},
    "fine_grained_pose": {"prefix": "nturgbd", "data_type": "video"},
    "character_order": {"prefix": "perception/videos", "data_type": "video"},
    "egocentric_navigation": {"prefix": "vlnqa", "data_type": "video"},
    "episodic_reasoning": {"prefix": "tvqa/frames_fps3_hq", "data_type": "frames", "frame_fps": 3.0},
    "counterfactual_inference": {"prefix": "clevrer/video_validation", "data_type": "video"},
}


def _format_options(candidates: Iterable[str]) -> str:
    """Format multiple-choice options in a stable A/B/C/... style."""

    return "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(candidates))


def _append_index(index: Dict[str, List[Path]], key: str, path: Path) -> None:
    """Store all candidates for one key so ambiguous names stay visible."""

    index.setdefault(key.replace("\\", "/"), []).append(path)


def _build_path_index(video_dir: Path) -> Dict[str, List[Path]]:
    """Index videos by suffix paths and filenames as a safe fallback resolver."""

    index: Dict[str, List[Path]] = {}
    for path in video_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        try:
            relative = path.relative_to(video_dir).as_posix()
        except ValueError:
            relative = path.as_posix()

        parts = Path(relative).parts
        _append_index(index, path.name, path)
        # Add every suffix, e.g. "left/a.mp4" can match "vlnqa/left/a.mp4".
        for start in range(len(parts)):
            _append_index(index, "/".join(parts[start:]), path)
    return index


def _selected_task_files(annotation_dir: Path, tasks: Any) -> List[Path]:
    """Resolve task JSON files from 'all', a string, or a list."""

    if tasks in (None, "all"):
        return sorted(annotation_dir.glob("*.json"))
    if isinstance(tasks, str):
        tasks = [tasks]
    return [annotation_dir / f"{task}.json" for task in tasks]


def _unique_index_match(index: Dict[str, List[Path]], key: str) -> Optional[Path]:
    """Return a fallback match only when the annotation key is unambiguous."""

    matches = index.get(key.replace("\\", "/"), [])
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_mvbench_path(video_dir: Path, task_name: str, raw_video: str, index: Dict[str, List[Path]]) -> Dict[str, Any]:
    """Resolve one MVBench media path and attach diagnostics for logging."""

    source = TASK_SOURCES.get(task_name, {})
    prefix = source.get("prefix")
    data_type = str(source.get("data_type", "video"))
    frame_fps = source.get("frame_fps")
    raw_key = raw_video.replace("\\", "/")

    if prefix:
        candidate = video_dir / prefix / raw_key
        # Frame-based tasks point to a directory, while normal tasks point to a
        # video file. Both can be consumed by video_io.load_video_frames.
        if candidate.exists():
            return {
                "path": candidate,
                "path_status": "task_mapping",
                "data_type": data_type,
                "frame_fps": frame_fps,
                "video_missing": False,
            }

    for key in [raw_key, Path(raw_key).name]:
        matched = _unique_index_match(index, key)
        if matched is not None:
            return {
                "path": matched,
                "path_status": "fallback_index",
                "data_type": "video",
                "frame_fps": frame_fps,
                "video_missing": False,
            }

    # Preserve the expected mapped location in the sample so skipped.jsonl tells
    # the user exactly which dataset asset is absent on the server.
    expected = video_dir / prefix / raw_key if prefix else video_dir / raw_key
    return {
        "path": expected,
        "path_status": "missing",
        "data_type": data_type,
        "frame_fps": frame_fps,
        "video_missing": True,
    }


def load_mvbench(config: Dict[str, Any]) -> List[EvalSample]:
    """Load MVBench annotations and return unified evaluation samples."""

    annotation_dir = Path(config["annotation_dir"])
    video_dir = Path(config["video_dir"])
    video_index = _build_path_index(video_dir)
    template = config.get("question_template", "{question}\n{options}")
    samples: List[EvalSample] = []

    for task_file in _selected_task_files(annotation_dir, config.get("tasks", "all")):
        task_name = task_file.stem
        records = json.loads(task_file.read_text(encoding="utf-8"))
        for idx, record in enumerate(records):
            choices = list(record.get("candidates", []))
            video_name = record["video"]
            resolved = _resolve_mvbench_path(video_dir, task_name, video_name, video_index)
            question = template.format(question=record["question"], options=_format_options(choices))
            metadata = {
                "raw_video": video_name,
                "resolved_video": str(resolved["path"]),
                "path_status": resolved["path_status"],
                "video_missing": resolved["video_missing"],
                "data_type": resolved["data_type"],
            }
            if resolved.get("frame_fps") is not None:
                metadata["frame_fps"] = float(resolved["frame_fps"])
            for key in ["start", "end", "start_time", "end_time"]:
                if key in record:
                    metadata[key] = record[key]
            samples.append(
                EvalSample(
                    sample_id=f"mvbench_{task_name}_{idx:05d}",
                    dataset="mvbench",
                    video_path=str(resolved["path"]),
                    question=question,
                    answer=str(record["answer"]),
                    choices=choices,
                    task=task_name,
                    metadata=metadata,
                )
            )
    return samples
