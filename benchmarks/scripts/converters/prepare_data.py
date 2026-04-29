"""Prepare synthetic benchmark smoke/dev data in the unified JSONL format."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.scripts.converters.synthetic import build_synthetic_examples
from benchmarks.scripts.schema import BenchmarkExample
from benchmarks.scripts.utils.io import write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]


def write_split(split: str) -> int:
    examples = build_synthetic_examples(split)
    rows_by_benchmark: dict[str, list[dict[str, object]]] = {}
    for example in examples:
        validated = BenchmarkExample.from_dict(example.to_dict())
        rows_by_benchmark.setdefault(validated.benchmark, []).append(validated.to_dict())

    for benchmark, rows in rows_by_benchmark.items():
        write_jsonl(REPO_ROOT / "benchmarks" / "data" / split / f"{benchmark}.jsonl", rows)
        write_jsonl(REPO_ROOT / "benchmarks" / "data" / "processed" / split / f"{benchmark}.jsonl", rows)
    return len(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare synthetic benchmark data")
    parser.add_argument("--splits", nargs="+", default=["smoke", "dev"], choices=["smoke", "dev"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        count = write_split(split)
        print(f"Wrote {count} {split} examples")


if __name__ == "__main__":
    main()
