"""Common helpers for offline official-data converters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmarks.scripts.schema import BenchmarkExample, SchemaError


AnswerTransform = Callable[[Any], str | None]


def first_present(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def stringify_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return stringify(value[0])
        return " | ".join(stringify(item) for item in value)
    return stringify(value)


def normalize_choices(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return [f"{key}: {stringify(choice)}" for key, choice in value.items()]
    if isinstance(value, list):
        return [stringify(choice) for choice in value]
    return [stringify(value)]


def context_length(text: str) -> int:
    return len(text.split())


def read_raw_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw data file does not exist: {input_path}")
    if input_path.suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise SchemaError(f"{input_path}:{line_number} must be a JSON object")
                rows.append(row)
        return rows
    if input_path.suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = first_present(payload, ["data", "examples", "instances", "rows"])
        else:
            rows = None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise SchemaError(f"{input_path} must contain a list of JSON objects")
        return rows
    raise ValueError(f"Unsupported raw data extension: {input_path.suffix}")


def require_text(row: dict[str, Any], keys: Iterable[str], *, field_name: str, row_id: str) -> str:
    value = first_present(row, keys)
    text = stringify(value).strip()
    if not text:
        raise SchemaError(f"Row {row_id} is missing required {field_name}; tried {list(keys)}")
    return text


def row_id(row: dict[str, Any], index: int, benchmark: str) -> str:
    value = first_present(row, ["id", "uid", "example_id", "qid", "question_id", "instance_id"])
    return str(value) if value is not None else f"{benchmark.lower()}-{index:06d}"


def build_example(
    *,
    row: dict[str, Any],
    index: int,
    benchmark: str,
    split: str,
    task: str,
    context: str,
    question: str,
    answer: str | None,
    metric: str,
    choices: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkExample:
    example_id = row_id(row, index, benchmark)
    example = BenchmarkExample(
        id=example_id,
        benchmark=benchmark,
        task=task,
        split=split,
        context=context,
        question=question,
        answer=answer,
        choices=choices,
        metadata={
            "context_length": context_length(context),
            "source": "official",
            "metric": metric,
            **(metadata or {}),
        },
    )
    return BenchmarkExample.from_dict(example.to_dict())

