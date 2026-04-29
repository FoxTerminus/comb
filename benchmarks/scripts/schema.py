"""Shared schemas for benchmark examples and generation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BENCHMARKS = {"RULER", "SCBench", "LongBench", "LoCoMo", "LongCodeBench"}
PRIMARY_BENCHMARKS = {"RULER", "SCBench", "LongBench"}
SECONDARY_BENCHMARKS = {"LoCoMo", "LongCodeBench"}
SPLITS = {"smoke", "dev", "full"}
METRICS = {"exact_match", "f1", "rouge_l", "classification", "code", "contains"}
KV_CACHE_POLICIES = {
    "mock_no_kv_cache",
    "full_decoder_kv_cache",
    "chunk_encoder_cross_attention_kv",
    "shared_cross_decoder_kv_cache",
    "hybrid_decoder_state_cache",
    "unknown",
}


class SchemaError(ValueError):
    """Raised when benchmark data violates the shared schema."""


def _require_string(row: dict[str, Any], key: str) -> str:
    if key not in row:
        raise SchemaError(f"Missing required field: {key}")
    value = row[key]
    if not isinstance(value, str):
        raise SchemaError(f"Field {key!r} must be a string, got {type(value).__name__}")
    return value


def _optional_string(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"Field {key!r} must be a string or null, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    benchmark: str
    task: str
    split: str
    context: str
    question: str
    answer: str | None = None
    choices: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BenchmarkExample":
        benchmark = _require_string(row, "benchmark")
        if benchmark not in BENCHMARKS:
            raise SchemaError(f"Unsupported benchmark {benchmark!r}; expected one of {sorted(BENCHMARKS)}")
        split = _require_string(row, "split")
        if split not in SPLITS:
            raise SchemaError(f"Unsupported split {split!r}; expected one of {sorted(SPLITS)}")

        choices = row.get("choices")
        if choices is not None:
            if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
                raise SchemaError("Field 'choices' must be a list of strings or null")

        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise SchemaError("Field 'metadata' must be an object")
        metric = metadata.get("metric")
        if metric is not None and metric not in METRICS:
            raise SchemaError(f"Unsupported metric {metric!r}; expected one of {sorted(METRICS)}")

        return cls(
            id=_require_string(row, "id"),
            benchmark=benchmark,
            task=_require_string(row, "task"),
            split=split,
            context=_require_string(row, "context"),
            question=_require_string(row, "question"),
            answer=_optional_string(row, "answer"),
            choices=choices,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRecord:
    run_id: str
    model: str
    checkpoint: str
    benchmark: str
    task: str
    id: str
    prediction: str
    answer: str | None
    metrics: dict[str, Any]
    example_metadata: dict[str, Any]
    prompt_tokens: int
    context_tokens: int
    generated_tokens: int
    kv_cache_policy: str
    chunk_size: int | None
    recent_window_tokens: int | None
    peak_memory_gb: float | None
    prefill_latency_s: float | None
    decode_latency_s: float | None
    tokens_per_second: float | None
    error: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GenerationRecord":
        for key in (
            "run_id",
            "model",
            "checkpoint",
            "benchmark",
            "task",
            "id",
            "prediction",
            "metrics",
            "example_metadata",
            "prompt_tokens",
            "context_tokens",
            "generated_tokens",
            "kv_cache_policy",
            "error",
        ):
            if key not in row:
                raise SchemaError(f"Missing required generation field: {key}")
        kv_cache_policy = row["kv_cache_policy"]
        if kv_cache_policy not in KV_CACHE_POLICIES:
            raise SchemaError(
                f"Unsupported kv_cache_policy {kv_cache_policy!r}; "
                f"expected one of {sorted(KV_CACHE_POLICIES)}"
            )
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_role(benchmark: str) -> str:
    if benchmark in PRIMARY_BENCHMARKS:
        return "primary"
    if benchmark in SECONDARY_BENCHMARKS:
        return "secondary"
    raise SchemaError(f"Unsupported benchmark {benchmark!r}")
