"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

from benchmarks.scripts.converters.prepare_data import write_split
from benchmarks.scripts.schema import BenchmarkExample
from benchmarks.scripts.utils.io import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_split_data(split: str) -> None:
    data_dir = REPO_ROOT / "benchmarks" / "data" / split
    expected = ["RULER", "SCBench", "LongBench", "LoCoMo", "LongCodeBench"]
    if all((data_dir / f"{benchmark}.jsonl").exists() for benchmark in expected):
        return
    if split in {"smoke", "dev"}:
        write_split(split)
        return
    raise FileNotFoundError(f"Missing full data in {data_dir}; run official converters first")


def load_examples(split: str, benchmarks: list[str]) -> list[BenchmarkExample]:
    ensure_split_data(split)
    examples: list[BenchmarkExample] = []
    for benchmark in benchmarks:
        path = REPO_ROOT / "benchmarks" / "data" / split / f"{benchmark}.jsonl"
        for row in read_jsonl(path):
            examples.append(BenchmarkExample.from_dict(row))
    return examples

