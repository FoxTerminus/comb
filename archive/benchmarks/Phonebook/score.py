"""Phonebook scoring."""

from __future__ import annotations

import re


def normalize_phone(text: str) -> str:
    return "".join(re.findall(r"\d", text or ""))


def score_prediction(prediction: str, answer: str) -> dict[str, float | str]:
    pred_digits = normalize_phone(prediction)
    answer_digits = normalize_phone(answer)
    exact = float(pred_digits == answer_digits)
    contains = float(bool(answer_digits) and answer_digits in pred_digits)
    return {
        "prediction_digits": pred_digits,
        "answer_digits": answer_digits,
        "exact": exact,
        "contains": contains,
    }
