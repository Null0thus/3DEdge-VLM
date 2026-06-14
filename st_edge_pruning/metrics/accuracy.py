from __future__ import annotations

import re
from typing import Iterable

OPTION_PREFIX_RE = re.compile(r"^\s*(?:option\s*)?\(?([a-z])\)?[\.\):]\s*(.*)$", re.IGNORECASE)
ANSWER_LETTER_RE = re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([a-z])\)?\b", re.IGNORECASE)
START_LETTER_RE = re.compile(r"^\s*\(?([a-z])\)?(?:\s*[\.\):]\s*|\s*$)", re.IGNORECASE)
ANSWER_TEXT_PREFIX_RE = re.compile(r"^\s*(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Normalize free-form model outputs for simple multiple-choice matching."""

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_option_prefix(text: str) -> str:
    """Remove leading option labels such as 'A.' or 'Option B:'."""

    match = OPTION_PREFIX_RE.match(str(text))
    if match:
        return match.group(2).strip()
    return str(text).strip()


def _strip_answer_text_prefix(text: str) -> str:
    """Remove short answer-introducing text before exact option matching."""

    stripped = _strip_option_prefix(text)
    return ANSWER_TEXT_PREFIX_RE.sub("", stripped).strip()


def _letter_to_index(letter: str, num_choices: int) -> int | None:
    """Map an option letter to an index only when it is valid."""

    index = ord(letter.lower()) - ord("a")
    if 0 <= index < num_choices:
        return index
    return None


def _answer_letter(answer: str, choices: list[str]) -> str | None:
    """Infer the gold option letter from a letter, labeled answer, or choice text."""

    raw_answer = str(answer).strip()
    start_match = START_LETTER_RE.match(raw_answer)
    if start_match and _letter_to_index(start_match.group(1), len(choices)) is not None:
        return start_match.group(1).upper()

    ans_norm = _normalize(_strip_option_prefix(raw_answer))
    for idx, choice in enumerate(choices):
        if _normalize(_strip_option_prefix(choice)) == ans_norm:
            return chr(65 + idx)
    return None


def _prediction_letter(prediction: str, choices: list[str]) -> str | None:
    """Extract an explicit predicted option letter without matching normal words."""

    stripped = str(prediction).strip()
    for pattern in (START_LETTER_RE, ANSWER_LETTER_RE):
        match = pattern.search(stripped)
        if match and _letter_to_index(match.group(1), len(choices)) is not None:
            return match.group(1).upper()
    return None


def _contains_non_gold_choice(pred_norm: str, choices: list[str], gold_index: int) -> bool:
    """Detect copied option lists so prompt echoes are not counted as answers."""

    for idx, choice in enumerate(choices):
        if idx == gold_index:
            continue
        choice_norm = _normalize(_strip_option_prefix(choice))
        if choice_norm and choice_norm in pred_norm:
            return True
    return False


def _matches_gold_text(prediction: str, choices: list[str], gold_index: int) -> bool:
    """Accept text answers only when the output itself is the gold option."""

    gold_text = _normalize(_strip_option_prefix(choices[gold_index]))
    if not gold_text:
        return False

    pred_norm = _normalize(prediction)
    if _contains_non_gold_choice(pred_norm, choices, gold_index):
        return False

    # Check both the raw output and a version with answer-introducing words
    # removed. This accepts "answer: dunking" but rejects copied option lists.
    candidates = [
        _normalize(_strip_option_prefix(prediction)),
        _normalize(_strip_answer_text_prefix(prediction)),
    ]
    return any(candidate == gold_text for candidate in candidates)


def is_correct(prediction: str, answer: str, choices: Iterable[str] | None = None) -> bool:
    """Check correctness by option letter or exact option text."""

    pred_norm = _normalize(prediction)
    if not pred_norm:
        return False

    choices = list(choices or [])
    gold_letter = _answer_letter(str(answer), choices) if choices else None
    if gold_letter is not None:
        pred_letter = _prediction_letter(prediction, choices)
        if pred_letter is not None:
            return pred_letter == gold_letter
        return _matches_gold_text(prediction, choices, ord(gold_letter) - ord("A"))

    ans_norm = _normalize(answer)
    if pred_norm.startswith(ans_norm) or ans_norm in pred_norm:
        return True
    return False
