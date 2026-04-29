"""Environment snapshot validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmarks.scripts.utils.config import repo_path
from benchmarks.scripts.utils.io import read_json, write_json


REQUIRED_ENVIRONMENT_KEYS = {
    "git_commit",
    "git_dirty",
    "git_status_short",
    "cuda_visible_devices",
    "cuda_available",
    "cuda_device_count",
    "torch_version",
    "transformers_version",
    "flash_attn_version",
    "hostname",
    "model_path",
    "tokenizer_path",
    "start_time",
    "end_time",
}


def validate_environment_snapshot(snapshot: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    missing = sorted(REQUIRED_ENVIRONMENT_KEYS - set(snapshot))
    for key in missing:
        problems.append(f"missing key: {key}")
    if snapshot.get("git_commit") is None:
        problems.append("git_commit is null; run may not be reproducible from git")
    if not isinstance(snapshot.get("git_dirty"), bool):
        problems.append("git_dirty must be a boolean")
    if not isinstance(snapshot.get("cuda_available"), bool):
        problems.append("cuda_available must be a boolean")
    if not isinstance(snapshot.get("cuda_device_count"), int):
        problems.append("cuda_device_count must be an integer")
    if snapshot.get("start_time") and snapshot.get("end_time") and snapshot["start_time"] > snapshot["end_time"]:
        problems.append("start_time is later than end_time")
    return problems


def validate_environment_file(path: str | Path) -> dict[str, Any]:
    snapshot_path = repo_path(path)
    snapshot = read_json(snapshot_path)
    problems = validate_environment_snapshot(snapshot)
    return {
        "path": str(snapshot_path),
        "ok": not problems,
        "problems": problems,
        "snapshot": snapshot,
    }


def write_environment_check(path: str | Path, output: str | Path | None = None) -> Path:
    result = validate_environment_file(path)
    output_path = repo_path(output) if output else repo_path(path).with_name("environment_check.json")
    write_json(output_path, result)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a benchmark environment.json snapshot")
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_environment_check(args.environment, args.output)
    result = read_json(output_path)
    print(f"Wrote environment check to {output_path}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
