"""Official-style LoCoMo QA scoring."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

try:
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover - only used in minimal envs.
    PorterStemmer = None


_STEMMER = PorterStemmer() if PorterStemmer else None


def normalize_answer(text: str) -> str:
    text = str(text).replace(",", "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stem(token: str) -> str:
    return _STEMMER.stem(token) if _STEMMER else token


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = [_stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [_stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in str(prediction).split(",") if p.strip()]
    golds = [g.strip() for g in str(ground_truth).split(",") if g.strip()]
    if not predictions:
        predictions = [str(prediction)]
    if not golds:
        golds = [str(ground_truth)]
    return mean(max(f1_score(pred, gold) for pred in predictions) for gold in golds)


def score_prediction(prediction: str, answer: str, category: int) -> float:
    if category in (2, 3, 4):
        gold = str(answer).split(";")[0].strip() if category == 3 else str(answer)
        return f1_score(prediction, gold)
    if category == 1:
        return multi_answer_f1(prediction, answer)
    if category == 5:
        pred = str(prediction).strip().lower()
        normalized = normalize_answer(pred)
        return 1.0 if (
            "no information available" in pred
            or "not mentioned" in pred
            or normalized in {"b", "option b", "choice b"}
        ) else 0.0
    raise ValueError(f"Unknown LoCoMo category: {category}")


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    scored = dict(record)
    scored["score"] = round(
        score_prediction(
            scored.get("prediction", ""),
            scored.get("answer", ""),
            int(scored["category"]),
        ),
        6,
    )
    return scored


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["model"])].append(rec)

    rows = []
    for model, items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "model": model,
            "n": len(items),
            "overall": mean(float(x["score"]) for x in items) if items else 0.0,
        }
        for category in range(1, 6):
            cat_items = [x for x in items if int(x["category"]) == category]
            row[f"cat{category}"] = (
                mean(float(x["score"]) for x in cat_items) if cat_items else 0.0
            )
        row["avg_latency_sec"] = mean(
            float(x.get("latency_sec", 0.0)) for x in items
        ) if items else 0.0
        row["avg_output_tokens"] = mean(
            int(x.get("output_tokens", 0)) for x in items
        ) if items else 0.0
        rows.append({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    return rows
