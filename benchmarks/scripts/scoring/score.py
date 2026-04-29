"""Score benchmark predictions and aggregate metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from benchmarks.scripts.schema import BenchmarkExample, GenerationRecord, benchmark_role
from benchmarks.scripts.scoring.metrics import (
    classification_accuracy,
    contains_match,
    edit_similarity,
    exact_match,
    rouge_l_f1,
    token_f1,
)

SCORE_KEYS = [
    "primary_score",
    "exact_match",
    "contains_match",
    "f1",
    "rouge_l",
    "classification_accuracy",
    "edit_similarity",
]


def score_prediction(prediction: str, answer: str | None, metric: str) -> dict[str, float | None]:
    scores = {
        "exact_match": exact_match(prediction, answer),
        "contains_match": contains_match(prediction, answer),
    }
    if metric == "f1":
        scores["f1"] = token_f1(prediction, answer)
    elif metric == "rouge_l":
        scores["rouge_l"] = rouge_l_f1(prediction, answer)
    elif metric == "classification":
        scores["classification_accuracy"] = classification_accuracy(prediction, answer)
    elif metric == "code":
        scores["edit_similarity"] = edit_similarity(prediction, answer)
    elif metric == "contains":
        scores["primary_score"] = scores["contains_match"]
    else:
        scores["primary_score"] = scores["exact_match"]

    if "primary_score" not in scores:
        preferred = {
            "f1": "f1",
            "rouge_l": "rouge_l",
            "classification": "classification_accuracy",
            "code": "edit_similarity",
        }.get(metric, "exact_match")
        scores["primary_score"] = scores.get(preferred)
    return scores


def attach_metrics(record: GenerationRecord, example: BenchmarkExample) -> GenerationRecord:
    if record.error is not None:
        record.metrics = {score_key: None for score_key in SCORE_KEYS}
        return record
    metric = str(example.metadata.get("metric", "exact_match"))
    record.metrics = score_prediction(record.prediction, record.answer, metric)
    return record


def summarize_records(records: list[GenerationRecord]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[GenerationRecord]] = defaultdict(list)
    for record in records:
        groups[("overall",)].append(record)
        groups[("benchmark", record.benchmark)].append(record)
        groups[("task", record.benchmark, record.task)].append(record)
        groups[("model", record.model)].append(record)
        groups[("kv_cache_policy", record.kv_cache_policy)].append(record)
        groups[("role", benchmark_role(record.benchmark))].append(record)
        groups[("length_bucket", _length_bucket(record.prompt_tokens))].append(record)
        for group in _metadata_groups(record):
            groups[group].append(record)

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        failures = [row for row in rows if row.error]
        score_means = {
            score_key: _mean_optional(
                row.metrics.get(score_key)
                for row in rows
                if row.error is None and row.metrics.get(score_key) is not None
            )
            for score_key in SCORE_KEYS
        }
        summaries.append(
            {
                "group": "/".join(key),
                "count": len(rows),
                "success_count": len(rows) - len(failures),
                "failure_count": len(failures),
                **score_means,
                "avg_peak_memory_gb": _mean_optional(row.peak_memory_gb for row in rows),
                "avg_prefill_latency_s": _mean_optional(row.prefill_latency_s for row in rows),
                "avg_decode_latency_s": _mean_optional(row.decode_latency_s for row in rows),
                "avg_tokens_per_second": _mean_optional(row.tokens_per_second for row in rows),
                "avg_prompt_tokens": _mean_optional(row.prompt_tokens for row in rows),
                "max_prompt_tokens": max((row.prompt_tokens for row in rows), default=None),
            }
        )
    return {"summaries": summaries, "record_count": len(records)}


def _mean_optional(values: object) -> float | None:
    usable = [float(value) for value in values if value is not None]  # type: ignore[union-attr]
    if not usable:
        return None
    return mean(usable)


def _length_bucket(prompt_tokens: int) -> str:
    for threshold in (4096, 8192, 16384, 32768, 65536, 131072):
        if prompt_tokens <= threshold:
            return f"<= {threshold}"
    return "> 131072"


def _metadata_groups(record: GenerationRecord) -> list[tuple[str, ...]]:
    metadata = record.example_metadata or {}
    groups: list[tuple[str, ...]] = []
    generic_keys = ("metric", "source")
    benchmark_keys = {
        "RULER": ("task_type", "needle_position", "target_context_length"),
        "SCBench": ("category", "cache_reuse"),
        "LongBench": ("official_task", "language", "length_bucket"),
        "LoCoMo": ("question_type", "temporal"),
        "LongCodeBench": ("task_type", "language"),
    }
    for key in generic_keys + benchmark_keys.get(record.benchmark, ()):
        value = metadata.get(key)
        if value is None or value == "":
            continue
        groups.append((f"metadata/{key}", _stringify_group_value(value)))
    return groups


def _stringify_group_value(value: object) -> str:
    text = str(value).replace("/", "_")
    if len(text) > 96:
        return text[:93] + "..."
    return text
