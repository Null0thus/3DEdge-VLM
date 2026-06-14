from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from st_edge_pruning.types import EvalSample

OPTION_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$")


def _parse_options(question: str) -> Tuple[str, List[str]]:
    """Split TempCompass multi-choice text into stem and option texts."""

    stem_lines: List[str] = []
    options: List[str] = []
    for line in str(question).splitlines():
        match = OPTION_RE.match(line)
        if match:
            options.append(match.group(2).strip())
        elif line.strip():
            stem_lines.append(line.strip())
    return "\n".join(stem_lines), options


def _parse_answer_letter(answer: str, choices: List[str]) -> str:
    """Extract an A/B/C answer from TempCompass' answer text."""

    match = OPTION_RE.match(str(answer))
    if match:
        return match.group(1)
    normalized_answer = str(answer).strip().lower()
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == normalized_answer:
            return chr(65 + idx)
    raise ValueError(f"Could not parse TempCompass answer: {answer!r}")


def _format_options(choices: List[str]) -> str:
    """Render parsed choices in a stable option-letter format."""

    return "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))


def _resolve_video_path(video_dir: Path, video_id: Any) -> Path:
    """Resolve one TempCompass video id to its local mp4 path."""

    return video_dir / f"{video_id}.mp4"


def load_tempcompass(config: Dict[str, Any]) -> List[EvalSample]:
    """Load TempCompass multi-choice/test as unified MC evaluation samples."""

    import pandas as pd

    annotation_file = Path(config["annotation_file"])
    video_dir = Path(config["video_dir"])
    template = config.get("question_template", "{question}\n{options}")
    frame = pd.read_parquet(annotation_file)
    samples: List[EvalSample] = []

    for idx, row in frame.iterrows():
        record = row.to_dict()
        stem, choices = _parse_options(str(record["question"]))
        if not choices:
            raise ValueError(f"TempCompass row has no parsed choices at index {idx}")
        answer = _parse_answer_letter(str(record["answer"]), choices)
        video_path = _resolve_video_path(video_dir, record["video_id"])
        question = template.format(question=stem, options=_format_options(choices))
        samples.append(
            EvalSample(
                sample_id=f"tempcompass_{record['video_id']}_{idx:05d}",
                dataset="tempcompass",
                video_path=str(video_path),
                question=question,
                answer=answer,
                choices=choices,
                task=str(record.get("dim", "")),
                metadata={
                    "raw_video": str(record["video_id"]),
                    "raw_question": str(record["question"]),
                    "raw_answer": str(record["answer"]),
                    "dim": str(record.get("dim", "")),
                    "row_index": int(idx),
                    "video_missing": not video_path.exists(),
                },
            )
        )
    return samples
