"""Build retry manifests from failed benchmark runs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.scripts.schema import GenerationRecord
from benchmarks.scripts.utils.config import repo_path
from benchmarks.scripts.utils.io import read_json, read_jsonl, write_json


def failure_ids(run_dir: str | Path) -> list[str]:
    predictions_path = repo_path(run_dir) / "predictions.jsonl"
    ids: list[str] = []
    for row in read_jsonl(predictions_path):
        record = GenerationRecord.from_dict(row)
        if record.error:
            ids.append(record.id)
    return ids


def build_retry_manifest(run_dir: str | Path) -> dict[str, Any]:
    run_path = repo_path(run_dir)
    records = [GenerationRecord.from_dict(row) for row in read_jsonl(run_path / "predictions.jsonl")]
    failed = [record for record in records if record.error]
    config = read_json(run_path / "run_config.resolved.json") if (run_path / "run_config.resolved.json").exists() else {}
    error_types = Counter(_error_type(record.error) for record in failed)
    return {
        "source_run_dir": str(run_path),
        "run_name": config.get("run_name"),
        "split": config.get("split"),
        "model": (config.get("model") or {}).get("name") if isinstance(config.get("model"), dict) else None,
        "total_records": len(records),
        "failed_count": len(failed),
        "ids": [record.id for record in failed],
        "error_types": dict(sorted(error_types.items())),
    }


def write_retry_manifest(run_dir: str | Path, output: str | Path | None = None) -> Path:
    run_path = repo_path(run_dir)
    output_path = repo_path(output) if output else run_path / "retry_ids.json"
    write_json(output_path, build_retry_manifest(run_path))
    return output_path


def _error_type(error: str | None) -> str:
    if not error:
        return "unknown"
    return error.split(":", 1)[0].strip() or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write retry_ids.json from failed predictions")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_retry_manifest(args.run_dir, args.output)
    print(f"Wrote retry manifest to {output_path}")


if __name__ == "__main__":
    main()
