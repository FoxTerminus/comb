"""Lightweight metrics for benchmark prediction summaries.

The implementations here intentionally avoid heavyweight benchmark packages so
that dev scoring can run offline on cached predictions. They mirror the metric
families used by the upstream benchmarks closely enough for phase gating, but
official full-report numbers should still be generated with official scripts
when those are integrated.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text.strip())
    return text


def normalize_answer(value: Any) -> str:
    """Normalize free-form answers following common QA evaluation practice."""

    text = normalize_text(value)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: Any, answer: Any) -> bool | None:
    if answer is None:
        return None
    return normalize_answer(prediction) == normalize_answer(answer)


def contains_match(prediction: Any, answer: Any) -> bool | None:
    if answer is None:
        return None
    pred = normalize_answer(prediction)
    gold = normalize_answer(answer)
    if not gold:
        return None
    return gold in pred or pred in gold


def token_f1(prediction: Any, answer: Any) -> float | None:
    if answer is None:
        return None
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: Any, answer: Any) -> float | None:
    """Compute Rouge-L F1 over normalized whitespace-delimited tokens."""

    if answer is None:
        return None
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    dp = [0] * (len(gold_tokens) + 1)
    for pred_token in pred_tokens:
        prev = 0
        for idx, gold_token in enumerate(gold_tokens, start=1):
            current = dp[idx]
            if pred_token == gold_token:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = current

    lcs = dp[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def edit_similarity(prediction: Any, answer: Any) -> float | None:
    """Return a 0-1 edit-similarity proxy for code completion tasks."""

    if answer is None:
        return None
    pred = "" if prediction is None else str(prediction).strip()
    gold = str(answer).strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return SequenceMatcher(None, pred, gold).ratio()
