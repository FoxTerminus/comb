"""Report generation for benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from benchmarks.scripts.schema import GenerationRecord
from benchmarks.scripts.scoring.score import summarize_records
from benchmarks.scripts.utils.io import write_json


def write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "count",
        "success_count",
        "failure_count",
        "primary_score",
        "exact_match",
        "contains_match",
        "f1",
        "rouge_l",
        "classification_accuracy",
        "edit_similarity",
        "avg_peak_memory_gb",
        "avg_prefill_latency_s",
        "avg_decode_latency_s",
        "avg_tokens_per_second",
        "avg_prompt_tokens",
        "max_prompt_tokens",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def write_failure_csv(path: str | Path, records: list[GenerationRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "benchmark", "task", "id", "error"])
        writer.writeheader()
        for record in records:
            if record.error:
                writer.writerow(
                    {
                        "model": record.model,
                        "benchmark": record.benchmark,
                        "task": record.task,
                        "id": record.id,
                        "error": record.error,
                    }
                )


def write_markdown_report(path: str | Path, *, title: str, metrics: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Total records: {metrics['record_count']}", ""]
    lines.append("| Group | Count | Success | Failure | Primary | EM | Contains | F1 | Rouge-L | Code/Edit |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in metrics["summaries"]:
        primary = _format_score(row.get("primary_score"))
        exact = _format_score(row.get("exact_match"))
        contains = _format_score(row.get("contains_match"))
        f1 = _format_score(row.get("f1"))
        rouge_l = _format_score(row.get("rouge_l"))
        edit = _format_score(row.get("edit_similarity"))
        lines.append(
            f"| {row['group']} | {row['count']} | {row['success_count']} | "
            f"{row['failure_count']} | {primary} | {exact} | {contains} | {f1} | {rouge_l} | {edit} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_run_reports(output_dir: str | Path, records: list[GenerationRecord], report_name: str) -> dict[str, Any]:
    output_path = Path(output_dir)
    metrics = summarize_records(records)
    write_json(output_path / "metrics.json", metrics)
    write_summary_csv(output_path / "benchmark_summary.csv", metrics["summaries"])
    write_failure_csv(output_path / "failures.csv", records)
    write_markdown_report(output_path / f"{report_name}.md", title=report_name.replace("_", " ").title(), metrics=metrics)
    return metrics


def _format_score(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"
