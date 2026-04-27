"""Shared schemas and JSONL helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    benchmark: str
    task: str
    context: str
    question: str
    answer: str | None = None
    choices: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BenchmarkExample":
        return cls(
            id=str(row["id"]),
            benchmark=str(row["benchmark"]),
            task=str(row["task"]),
            context=str(row.get("context", "")),
            question=str(row.get("question", "")),
            answer=None if row.get("answer") is None else str(row.get("answer")),
            choices=None if row.get("choices") is None else [str(choice) for choice in row["choices"]],
            metadata=row.get("metadata"),
        )


@dataclass
class GenerationRecord:
    id: str
    benchmark: str
    task: str
    model: str
    prompt_tokens: int
    chunk_tokens: int
    decoder_tokens: int
    generated_tokens: int
    compression_ratio: float
    prediction: str
    answer: str | None
    exact_match: bool | None
    peak_memory_gb: float | None
    prefill_latency_s: float
    decode_latency_s: float
    tokens_per_second: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
