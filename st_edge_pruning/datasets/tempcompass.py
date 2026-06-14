from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from st_edge_pruning.types import EvalSample

OPTION_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$")
CAPTION_MATCH_RE = re.compile(r"^\s*(Option|Sentence|Caption)\s+([A-Z0-9]+)\s*:\s*(.+?)\s*$", re.IGNORECASE)


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


def _caption_label_to_letter(label: str) -> str:
    """Map TempCompass caption-matching labels such as '1' or 'B' to A/B."""

    label = str(label).strip().upper()
    if label.isdigit():
        return chr(64 + int(label))
    return label


def _parse_caption_matching_options(question: str) -> Tuple[str, List[str]]:
    """Split TempCompass caption-matching text into a normal option list."""

    stem_lines: List[str] = []
    labeled_choices: List[Tuple[str, str]] = []
    for line in str(question).splitlines():
        match = CAPTION_MATCH_RE.match(line)
        if match:
            letter = _caption_label_to_letter(match.group(2))
            labeled_choices.append((letter, match.group(3).strip()))
        elif line.strip():
            stem_lines.append(line.strip())

    # The dataset uses Option 1/2, Sentence A/B, or Caption A/B. Sorting by the
    # normalized A/B label keeps the prompt order stable across all variants.
    labeled_choices.sort(key=lambda item: item[0])
    return "\n".join(stem_lines), [choice for _letter, choice in labeled_choices]


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


def _parse_caption_matching_answer(answer: str, choices: List[str]) -> str:
    """Extract an A/B answer from caption-matching labels or exact text."""

    match = CAPTION_MATCH_RE.match(str(answer))
    if match:
        return _caption_label_to_letter(match.group(2))
    normalized_answer = str(answer).strip().lower()
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == normalized_answer:
            return chr(65 + idx)
    raise ValueError(f"Could not parse TempCompass caption-matching answer: {answer!r}")


def _format_options(choices: List[str]) -> str:
    """Render parsed choices in a stable option-letter format."""

    return "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))


def _resolve_video_path(video_dir: Path, video_id: Any) -> Path:
    """Resolve one TempCompass video id to its local mp4 path."""

    return video_dir / f"{video_id}.mp4"


def _make_sample(
    record: Dict[str, Any],
    idx: int,
    video_dir: Path,
    template: str,
    question_format: str,
    stem: str,
    choices: List[str],
    answer: str,
) -> EvalSample:
    """Create one unified sample while preserving TempCompass diagnostics."""

    video_path = _resolve_video_path(video_dir, record["video_id"])
    question = template.format(question=stem, options=_format_options(choices))
    return EvalSample(
        sample_id=f"tempcompass_{question_format}_{record['video_id']}_{idx:05d}",
        dataset="tempcompass",
        video_path=str(video_path),
        question=question,
        answer=answer,
        choices=choices,
        task=f"{question_format}/{record.get('dim', '')}",
        metadata={
            "raw_video": str(record["video_id"]),
            "raw_question": str(record["question"]),
            "raw_answer": str(record["answer"]),
            "dim": str(record.get("dim", "")),
            "question_format": question_format,
            "row_index": int(idx),
            "video_missing": not video_path.exists(),
        },
    )


def _load_multi_choice(annotation_file: Path, video_dir: Path, template: str) -> List[EvalSample]:
    """Load TempCompass multi-choice questions as A/B/C/... samples."""

    import pandas as pd

    frame = pd.read_parquet(annotation_file)
    samples: List[EvalSample] = []
    for idx, row in frame.iterrows():
        record = row.to_dict()
        stem, choices = _parse_options(str(record["question"]))
        if not choices:
            raise ValueError(f"TempCompass multi-choice row has no parsed choices at index {idx}")
        answer = _parse_answer_letter(str(record["answer"]), choices)
        samples.append(_make_sample(record, idx, video_dir, template, "multi_choice", stem, choices, answer))
    return samples


def _load_caption_matching(annotation_file: Path, video_dir: Path, template: str) -> List[EvalSample]:
    """Load TempCompass caption-matching questions as stable A/B samples."""

    import pandas as pd

    frame = pd.read_parquet(annotation_file)
    samples: List[EvalSample] = []
    for idx, row in frame.iterrows():
        record = row.to_dict()
        stem, choices = _parse_caption_matching_options(str(record["question"]))
        if not choices:
            raise ValueError(f"TempCompass caption-matching row has no parsed choices at index {idx}")
        answer = _parse_caption_matching_answer(str(record["answer"]), choices)
        samples.append(_make_sample(record, idx, video_dir, template, "caption_matching", stem, choices, answer))
    return samples


def load_tempcompass(config: Dict[str, Any]) -> List[EvalSample]:
    """Load stable TempCompass multiple-choice formats for evaluation."""

    annotation_file = Path(config["annotation_file"])
    video_dir = Path(config["video_dir"])
    template = config.get("question_template", "{question}\n{options}")
    samples = _load_multi_choice(annotation_file, video_dir, template)

    # Caption matching is also a closed-set multiple-choice task. The unstable
    # yes/no and captioning formats stay excluded unless explicitly added later.
    if bool(config.get("include_caption_matching", False)):
        caption_matching_file = Path(config["caption_matching_file"])
        samples.extend(_load_caption_matching(caption_matching_file, video_dir, template))
    return samples
