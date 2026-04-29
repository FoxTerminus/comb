"""Failure summaries for benchmark runs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.scripts.schema import GenerationRecord
from benchmarks.scripts.utils.config import repo_path
from benchmarks.scripts.utils.io import read_jsonl, write_json


def summarize_failures(run_dir: str | Path) -> dict[str, Any]:
    run_path = repo_path(run_dir)
    records = [GenerationRecord.from_dict(row) for row in read_jsonl(run_path / "predictions.jsonl")]
    failed = [record for record in records if record.error]
    by_error_type = Counter(_error_type(record.error) for record in failed)
    by_model = Counter(record.model for record in failed)
    by_benchmark = Counter(record.benchmark for record in failed)
    by_task = Counter(f"{record.benchmark}/{record.task}" for record in failed)
    return {
        "run_dir": str(run_path),
        "total_records": len(records),
        "failed_count": len(failed),
        "success_count": len(records) - len(failed),
        "by_error_type": dict(sorted(by_error_type.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "by_task": dict(sorted(by_task.items())),
        "failures": [
            {
                "model": record.model,
                "benchmark": record.benchmark,
                "task": record.task,
                "id": record.id,
                "error_type": _error_type(record.error),
                "error": record.error,
            }
            for record in failed
        ],
    }


def write_failure_artifacts(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    run_path = repo_path(run_dir)
    target = repo_path(output_dir) if output_dir else run_path
    target.mkdir(parents=True, exist_ok=True)
    summary = summarize_failures(run_path)
    json_path = target / "failure_summary.json"
    csv_path = target / "failure_summary.csv"
    md_path = target / "failure_summary.md"
    write_json(json_path, summary)
    _write_failure_csv(csv_path, summary["failures"])
    _write_failure_markdown(md_path, summary)
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _write_failure_csv(path: Path, failures: list[dict[str, Any]]) -> None:
    fieldnames = ["model", "benchmark", "task", "id", "error_type", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failures:
            writer.writerow(row)


def _write_failure_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Failure Summary",
        "",
        f"Total records: {summary['total_records']}",
        f"Failed records: {summary['failed_count']}",
        f"Successful records: {summary['success_count']}",
        "",
        "## Error Types",
        "",
    ]
    if summary["by_error_type"]:
        for name, count in summary["by_error_type"].items():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No failures.")
    lines.extend(["", "## Failed Samples", ""])
    if summary["failures"]:
        lines.append("| Model | Benchmark | Task | ID | Error Type |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in summary["failures"]:
            lines.append(
                f"| {row['model']} | {row['benchmark']} | {row['task']} | "
                f"{row['id']} | {row['error_type']} |"
            )
    else:
        lines.append("No failed samples.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _error_type(error: str | None) -> str:
    if not error:
        return "unknown"
    return error.split(":", 1)[0].strip() or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize failed predictions from a benchmark run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_failure_artifacts(args.run_dir, args.output_dir)
    print(f"Wrote failure artifacts to {outputs['json'].parent}")


if __name__ == "__main__":
    main()
