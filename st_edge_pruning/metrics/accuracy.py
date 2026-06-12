from __future__ import annotations

import re
from typing import Iterable


def _normalize(text: str) -> str:
    """Normalize free-form model outputs for simple multiple-choice matching."""

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct(prediction: str, answer: str, choices: Iterable[str] | None = None) -> bool:
    """Check correctness by option letter or exact option text."""

    pred_norm = _normalize(prediction)
    ans_norm = _normalize(answer)
    if not pred_norm:
        return False
    if pred_norm.startswith(ans_norm) or ans_norm in pred_norm:
        return True
    choices = list(choices or [])
    for idx, choice in enumerate(choices):
        letter = chr(65 + idx).lower()
        if _normalize(choice) == ans_norm:
            # Match "A", "A.", or the option text.
            return pred_norm.startswith(letter) or _normalize(choice) in pred_norm
    return False
