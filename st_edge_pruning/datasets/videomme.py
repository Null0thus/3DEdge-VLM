from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from st_edge_pruning.types import EvalSample


def _format_options(options: Iterable[str]) -> str:
    """Format Video-MME options while preserving their original letters."""

    return "\n".join(str(option) for option in options)


def _resolve_video_path(video_dir: Path, record: Dict[str, Any]) -> Path:
    """Resolve Video-MME video paths by YouTube id first, then video id."""

    candidates = [
        video_dir / f"{record.get('videoID')}.mp4",
        video_dir / f"{record.get('video_id')}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_videomme(config: Dict[str, Any]) -> List[EvalSample]:
    """Load Video-MME parquet annotations and return unified samples."""

    import pandas as pd

    annotation_file = Path(config["annotation_file"])
    video_dir = Path(config["video_dir"])
    template = config.get("question_template", "{question}\n{options}")
    frame = pd.read_parquet(annotation_file)
    samples: List[EvalSample] = []

    for idx, row in frame.iterrows():
        record = row.to_dict()
        options = [str(x) for x in list(record["options"])]
        question = template.format(question=record["question"], options=_format_options(options))
        samples.append(
            EvalSample(
                sample_id=f"videomme_{record['question_id']}",
                dataset="videomme",
                video_path=str(_resolve_video_path(video_dir, record)),
                question=question,
                answer=str(record["answer"]),
                choices=options,
                task=str(record.get("task_type", "")),
                metadata={
                    "video_id": record.get("video_id"),
                    "videoID": record.get("videoID"),
                    "duration": record.get("duration"),
                    "domain": record.get("domain"),
                    "sub_category": record.get("sub_category"),
                    "row_index": int(idx),
                },
            )
        )
    return samples
