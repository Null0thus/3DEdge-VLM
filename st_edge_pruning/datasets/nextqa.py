from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from st_edge_pruning.types import EvalSample


def _format_options(options: Iterable[str]) -> str:
    """Format NExT-QA's a0-a4 fields as A/B/C/D/E choices."""

    return "\n".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))


def _answer_letter(answer_index: Any, num_options: int) -> str:
    """Convert the original NExT-QA 0-4 answer id into an option letter."""

    index = int(answer_index)
    if index < 0 or index >= num_options:
        raise ValueError(f"NExT-QA answer index out of range: {answer_index!r}")
    return chr(65 + index)


def _resolve_video_path(video_dir: Path, video_id: Any) -> Path:
    """Resolve one NExT-QA video id to its local mp4 path."""

    return video_dir / f"{video_id}.mp4"


def load_nextqa(config: Dict[str, Any]) -> List[EvalSample]:
    """Load NExT-QA MC/test annotations as unified multiple-choice samples."""

    import pandas as pd

    annotation_file = Path(config["annotation_file"])
    video_dir = Path(config["video_dir"])
    template = config.get("question_template", "{question}\n{options}")
    frame = pd.read_parquet(annotation_file)
    samples: List[EvalSample] = []

    for idx, row in frame.iterrows():
        record = row.to_dict()
        choices = [str(record[f"a{choice_idx}"]) for choice_idx in range(5)]
        answer_index = int(record["answer"])
        answer = _answer_letter(answer_index, len(choices))
        video_path = _resolve_video_path(video_dir, record["video"])
        question = template.format(question=record["question"], options=_format_options(choices))
        samples.append(
            EvalSample(
                sample_id=f"nextqa_{int(record['qid']):05d}_{idx:05d}",
                dataset="nextqa",
                video_path=str(video_path),
                question=question,
                answer=answer,
                choices=choices,
                task=str(record.get("type", "")),
                metadata={
                    "raw_video": str(record["video"]),
                    "raw_question": str(record["question"]),
                    "raw_answer": int(record["answer"]),
                    "answer_index": answer_index,
                    "frame_count": int(record.get("frame_count", 0)),
                    "width": int(record.get("width", 0)),
                    "height": int(record.get("height", 0)),
                    "row_index": int(idx),
                    "video_missing": not video_path.exists(),
                },
            )
        )
    return samples
