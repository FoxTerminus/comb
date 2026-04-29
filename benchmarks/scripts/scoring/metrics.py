"""Lightweight scoring metrics for benchmark predictions."""

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
    text = normalize_text(value)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: Any, answer: Any) -> float | None:
    if answer is None:
        return None
    return float(normalize_answer(prediction) == normalize_answer(answer))


def contains_match(prediction: Any, answer: Any) -> float | None:
    if answer is None:
        return None
    pred = normalize_answer(prediction)
    gold = normalize_answer(answer)
    if not gold:
        return None
    if not pred:
        return 0.0
    return float(gold in pred)


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
    if answer is None:
        return None
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    dp = [0] * (len(gold_tokens) + 1)
    for pred_token in pred_tokens:
        prev = 0
        for index, gold_token in enumerate(gold_tokens, start=1):
            current = dp[index]
            if pred_token == gold_token:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = current
    lcs = dp[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def edit_similarity(prediction: Any, answer: Any) -> float | None:
    if answer is None:
        return None
    pred = "" if prediction is None else str(prediction).strip()
    gold = str(answer).strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return SequenceMatcher(None, pred, gold).ratio()


def classification_accuracy(prediction: Any, answer: Any) -> float | None:
    return exact_match(prediction, answer)
