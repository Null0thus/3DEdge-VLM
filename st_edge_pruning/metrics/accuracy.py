from __future__ import annotations

import re
from typing import Iterable, List, Optional


def _normalize(text: str) -> str:
    """Normalize free-form model outputs for simple multiple-choice matching."""

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _answer_letter(answer: str, choices: List[str]) -> Optional[str]:
    """Map a dataset answer to its multiple-choice letter when possible."""

    ans_norm = _normalize(answer)
    if re.fullmatch(r"[a-z]", ans_norm):
        return ans_norm.upper()
    for idx, choice in enumerate(choices):
        if _normalize(choice) == ans_norm:
            return chr(65 + idx)
    return None


def _extract_option_letters(prediction: str, num_choices: int) -> List[str]:
    """Extract explicit option letters such as 'C', '(C)', or 'Option C'."""

    if num_choices <= 0:
        return []
    max_letter = chr(64 + num_choices)
    letters = []
    # Require letter boundaries so normal words do not create false matches.
    pattern = rf"(?<![A-Za-z])([A-{max_letter}a-{max_letter.lower()}])(?![A-Za-z])"
    for match in re.finditer(pattern, prediction):
        letters.append(match.group(1).upper())
    return letters


def _contains_choice_text(pred_norm: str, choice_norm: str) -> bool:
    """Match a full normalized choice phrase without arbitrary substring bugs."""

    if not choice_norm:
        return False
    # Single-token answers like "no" must be matched as a whole token, otherwise
    # "not sure" incorrectly contains "no".
    return re.search(rf"(^| ){re.escape(choice_norm)}($| )", pred_norm) is not None


def is_correct(prediction: str, answer: str, choices: Iterable[str] | None = None) -> bool:
    """Check MVBench-style multiple-choice answers robustly."""

    pred_norm = _normalize(prediction)
    ans_norm = _normalize(answer)
    if not pred_norm:
        return False
    if pred_norm.startswith(ans_norm) or ans_norm in pred_norm:
        # Only allow this shortcut for multi-token free-form answers. Single
        # tokens are handled with boundary-aware matching below.
        if " " in ans_norm:
            return True
    choices = list(choices or [])
    answer_letter = _answer_letter(answer, choices)
    if answer_letter is not None:
        extracted = _extract_option_letters(prediction, len(choices))
        if extracted:
            # Official MVBench-style evaluation scores the model's explicit
            # option letter. Use the first explicit letter as the final answer.
            return extracted[0] == answer_letter

    for choice in choices:
        choice_norm = _normalize(choice)
        if choice_norm == ans_norm:
            return _contains_choice_text(pred_norm, choice_norm)

    return _contains_choice_text(pred_norm, ans_norm)
