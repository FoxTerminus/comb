"""Validate converted benchmark data without running models."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.scripts.schema import BENCHMARKS, BenchmarkExample
from benchmarks.scripts.utils.io import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]


def validate_file(path: Path) -> int:
    rows = read_jsonl(path)
    ids: set[str] = set()
    for row in rows:
        example = BenchmarkExample.from_dict(row)
        if example.id in ids:
            raise ValueError(f"Duplicate id {example.id!r} in {path}")
        ids.add(example.id)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed benchmark JSONL files")
    parser.add_argument("--split", default="dev", choices=["smoke", "dev", "full"])
    parser.add_argument("--benchmarks", nargs="+", default=sorted(BENCHMARKS), choices=sorted(BENCHMARKS))
    parser.add_argument("--data-root", default="benchmarks/data/processed")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    total = 0
    for benchmark in args.benchmarks:
        path = data_root / args.split / f"{benchmark}.jsonl"
        if not path.exists():
            if args.allow_missing:
                print(f"SKIP {benchmark}: missing {path}")
                continue
            raise FileNotFoundError(f"Missing processed file: {path}")
        count = validate_file(path)
        total += count
        print(f"OK {benchmark}/{args.split}: {count} examples")
    print(f"Validated {total} examples")


if __name__ == "__main__":
    main()

