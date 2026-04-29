"""Offline converters for official benchmark rows.

These converters intentionally operate on local raw JSON/JSONL files only. They
standardize common official-dataset field names into the shared benchmark
schema, while preserving benchmark-specific metadata needed by reports.
"""

from __future__ import annotations

from typing import Any

from benchmarks.scripts.converters.common import (
    build_example,
    first_present,
    normalize_choices,
    require_text,
    stringify,
    stringify_answer,
)
from benchmarks.scripts.schema import BenchmarkExample


def convert_ruler(rows: list[dict[str, Any]], split: str) -> list[BenchmarkExample]:
    examples = []
    for index, row in enumerate(rows):
        rid = str(first_present(row, ["id", "uid"], f"ruler-{index:06d}"))
        task = stringify(first_present(row, ["task", "task_type", "name"], "unknown")).strip()
        context = require_text(row, ["context", "input", "document", "passage"], field_name="context", row_id=rid)
        question = require_text(row, ["question", "query", "instruction", "prompt"], field_name="question", row_id=rid)
        answer = stringify_answer(first_present(row, ["answer", "answers", "target", "gold", "label"]))
        examples.append(
            build_example(
                row=row,
                index=index,
                benchmark="RULER",
                split=split,
                task=task,
                context=context,
                question=question,
                answer=answer,
                metric=stringify(first_present(row, ["metric"], "exact_match")),
                metadata={
                    "task_type": stringify(first_present(row, ["task_type", "category"], task)),
                    "needle_position": first_present(row, ["needle_position", "answer_position", "position"]),
                    "target_context_length": first_present(row, ["context_length", "length", "tokens"]),
                },
            )
        )
    return examples


def convert_scbench(rows: list[dict[str, Any]], split: str) -> list[BenchmarkExample]:
    examples = []
    for index, row in enumerate(rows):
        rid = str(first_present(row, ["id", "uid"], f"scbench-{index:06d}"))
        task = stringify(first_present(row, ["task", "category", "type"], "unknown")).strip()
        shared_context = first_present(row, ["shared_context", "context", "input", "document"])
        context = require_text({"context": shared_context}, ["context"], field_name="context", row_id=rid)
        question = require_text(row, ["question", "query", "instruction", "prompt"], field_name="question", row_id=rid)
        examples.append(
            build_example(
                row=row,
                index=index,
                benchmark="SCBench",
                split=split,
                task=task,
                context=context,
                question=question,
                answer=stringify_answer(first_present(row, ["answer", "answers", "target", "gold", "label"])),
                metric=stringify(first_present(row, ["metric"], "contains")),
                choices=normalize_choices(first_present(row, ["choices", "options"])),
                metadata={
                    "shared_context_id": first_present(row, ["shared_context_id", "context_id", "group_id"]),
                    "query_index": first_present(row, ["query_index", "turn", "idx"]),
                    "cache_reuse": first_present(row, ["cache_reuse", "reuse"], True),
                    "category": first_present(row, ["category", "task_type"], task),
                },
            )
        )
    return examples


def convert_longbench(rows: list[dict[str, Any]], split: str) -> list[BenchmarkExample]:
    examples = []
    for index, row in enumerate(rows):
        rid = str(first_present(row, ["id", "uid"], f"longbench-{index:06d}"))
        official_task = stringify(first_present(row, ["dataset", "task", "official_task", "subset"], "unknown"))
        context = require_text(row, ["context", "input", "document", "passage"], field_name="context", row_id=rid)
        question = require_text(row, ["question", "query", "instruction", "prompt"], field_name="question", row_id=rid)
        metric = stringify(first_present(row, ["metric"], _longbench_default_metric(official_task)))
        examples.append(
            build_example(
                row=row,
                index=index,
                benchmark="LongBench",
                split=split,
                task=official_task,
                context=context,
                question=question,
                answer=stringify_answer(first_present(row, ["answer", "answers", "target", "gold", "label"])),
                metric=metric,
                choices=normalize_choices(first_present(row, ["choices", "options", "all_classes"])),
                metadata={
                    "official_task": official_task,
                    "language": first_present(row, ["language", "lang"], "unknown"),
                    "length_bucket": first_present(row, ["length_bucket", "length_range"]),
                },
            )
        )
    return examples


def convert_locomo(rows: list[dict[str, Any]], split: str) -> list[BenchmarkExample]:
    examples = []
    for index, row in enumerate(rows):
        rid = str(first_present(row, ["id", "uid", "question_id"], f"locomo-{index:06d}"))
        context_value = first_present(row, ["context", "conversation", "dialogue", "sessions", "history"])
        context = _render_dialogue(context_value)
        if not context:
            raise ValueError(f"Row {rid} is missing LoCoMo dialogue context")
        question = require_text(row, ["question", "query", "instruction", "prompt"], field_name="question", row_id=rid)
        examples.append(
            build_example(
                row=row,
                index=index,
                benchmark="LoCoMo",
                split=split,
                task=stringify(first_present(row, ["task"], "text_qa")),
                context=context,
                question=question,
                answer=stringify_answer(first_present(row, ["answer", "answers", "target", "gold", "label"])),
                metric=stringify(first_present(row, ["metric"], "f1")),
                choices=normalize_choices(first_present(row, ["choices", "options"])),
                metadata={
                    "session_id": first_present(row, ["session_id", "conversation_id", "dialogue_id"]),
                    "turn": first_present(row, ["turn", "turn_id"]),
                    "question_type": first_present(row, ["question_type", "type", "category"]),
                    "temporal": first_present(row, ["temporal", "is_temporal"], False),
                },
            )
        )
    return examples


def convert_longcodebench(rows: list[dict[str, Any]], split: str) -> list[BenchmarkExample]:
    examples = []
    for index, row in enumerate(rows):
        rid = str(first_present(row, ["id", "uid", "task_id"], f"longcodebench-{index:06d}"))
        context = require_text(
            row,
            ["context", "code_context", "repository", "repo_context", "input", "prompt"],
            field_name="code context",
            row_id=rid,
        )
        question = stringify(first_present(row, ["question", "instruction", "query"], "")).strip()
        if not question:
            question = "Complete or answer the code task using the repository context."
        task = stringify(first_present(row, ["task", "task_type", "category"], "code_qa"))
        examples.append(
            build_example(
                row=row,
                index=index,
                benchmark="LongCodeBench",
                split=split,
                task=task,
                context=context,
                question=question,
                answer=stringify_answer(first_present(row, ["answer", "answers", "target", "gold", "label", "completion"])),
                metric=stringify(first_present(row, ["metric"], "code")),
                metadata={
                    "repo": first_present(row, ["repo", "repository_name", "project"]),
                    "file": first_present(row, ["file", "filepath", "path"]),
                    "language": first_present(row, ["language", "lang"]),
                    "task_type": first_present(row, ["task_type", "category"], task),
                },
            )
        )
    return examples


CONVERTERS = {
    "RULER": convert_ruler,
    "SCBench": convert_scbench,
    "LongBench": convert_longbench,
    "LoCoMo": convert_locomo,
    "LongCodeBench": convert_longcodebench,
}


def _longbench_default_metric(task: str) -> str:
    lowered = task.lower()
    if any(name in lowered for name in ["trec", "lsht", "classification"]):
        return "classification"
    if any(name in lowered for name in ["summ", "gov_report", "qmsum", "multi_news"]):
        return "rouge_l"
    if "code" in lowered:
        return "code"
    return "f1"


def _render_dialogue(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                speaker = first_present(item, ["speaker", "role", "agent"], "speaker")
                text = first_present(item, ["text", "content", "utterance", "message"], "")
                lines.append(f"{speaker}: {stringify(text)}")
            else:
                lines.append(stringify(item))
        return "\n".join(lines)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {stringify(item)}" for key, item in value.items())
    return stringify(value)

