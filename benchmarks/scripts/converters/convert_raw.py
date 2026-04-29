"""Convert local raw benchmark files into the unified schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.scripts.converters.common import read_raw_rows
from benchmarks.scripts.converters.official import CONVERTERS
from benchmarks.scripts.schema import BENCHMARKS, BenchmarkExample
from benchmarks.scripts.utils.io import write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]


def default_raw_path(benchmark: str, split: str) -> Path:
    base = REPO_ROOT / "benchmarks" / "data" / "raw" / benchmark
    jsonl = base / f"{split}.jsonl"
    if jsonl.exists():
        return jsonl
    return base / f"{split}.json"


def convert_file(benchmark: str, split: str, raw_path: Path, output_root: Path) -> list[BenchmarkExample]:
    if benchmark not in CONVERTERS:
        raise ValueError(f"Unsupported benchmark converter: {benchmark}")
    rows = read_raw_rows(raw_path)
    examples = CONVERTERS[benchmark](rows, split)
    for example in examples:
        BenchmarkExample.from_dict(example.to_dict())

    rows_out = [example.to_dict() for example in examples]
    write_jsonl(output_root / split / f"{benchmark}.jsonl", rows_out)
    write_json(
        output_root / split / f"{benchmark}.manifest.json",
        {
            "benchmark": benchmark,
            "split": split,
            "raw_path": str(raw_path),
            "num_examples": len(examples),
            "source": "official",
        },
    )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local raw official benchmark data")
    parser.add_argument("--benchmarks", nargs="+", default=sorted(BENCHMARKS), choices=sorted(BENCHMARKS))
    parser.add_argument("--split", default="dev", choices=["dev", "full"])
    parser.add_argument("--raw-path", default=None, help="Single raw file path; only valid with one benchmark")
    parser.add_argument("--output-root", default="benchmarks/data/processed")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if args.raw_path and len(args.benchmarks) != 1:
        raise SystemExit("--raw-path can only be used with a single benchmark")

    total = 0
    for benchmark in args.benchmarks:
        raw_path = Path(args.raw_path) if args.raw_path else default_raw_path(benchmark, args.split)
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path
        if not raw_path.exists():
            if args.allow_missing:
                print(f"SKIP {benchmark}: missing {raw_path}")
                continue
            raise FileNotFoundError(f"Missing raw file for {benchmark}/{args.split}: {raw_path}")
        examples = convert_file(benchmark, args.split, raw_path, output_root)
        total += len(examples)
        print(f"Wrote {len(examples)} {benchmark}/{args.split} examples")
    print(f"Converted {total} examples")


if __name__ == "__main__":
    main()
