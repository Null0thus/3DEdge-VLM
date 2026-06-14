from __future__ import annotations

import re
from typing import Iterable

OPTION_PREFIX_RE = re.compile(r"^\s*(?:option\s*)?\(?([a-z])\)?[\.\):]\s*(.*)$", re.IGNORECASE)
ANSWER_LETTER_RE = re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([a-z])\)?\b", re.IGNORECASE)
START_LETTER_RE = re.compile(r"^\s*\(?([a-z])\)?(?:\s*[\.\):]\s*|\s*$)", re.IGNORECASE)


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
        gold_choice = choices[ord(gold_letter) - ord("A")]
        gold_text = _normalize(_strip_option_prefix(gold_choice))
        # If the model writes the answer text instead of a letter, accept only
        # the gold option text, not arbitrary single-letter substrings.
        return bool(gold_text) and (pred_norm.startswith(gold_text) or gold_text in pred_norm)

    ans_norm = _normalize(answer)
    if pred_norm.startswith(ans_norm) or ans_norm in pred_norm:
        return True
    return False
