#!/usr/bin/env python
"""Collect Phonebook predictions into score tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data3/junhaohu/comb/benchmarks/Phonebook/results/phonebook_32k")
    return parser.parse_args()


def bucket(position_frac: float) -> str:
    if position_frac < 1 / 3:
        return "early"
    if position_frac < 2 / 3:
        return "middle"
    return "late"


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in rows:
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                value = round(value, 6)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = []
    for path in sorted((results_dir / "predictions").glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"No prediction jsonl files under {results_dir / 'predictions'}")

    grouped = defaultdict(list)
    by_bucket = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
        by_bucket[(row["model"], bucket(float(row["target_position_frac"])))].append(row)

    summary_rows = []
    for model, parts in sorted(grouped.items()):
        summary_rows.append(
            {
                "model": model,
                "num_samples": len(parts),
                "exact": mean([float(r.get("exact", 0.0)) for r in parts]),
                "contains": mean([float(r.get("contains", 0.0)) for r in parts]),
                "avg_input_tokens": mean([float(r.get("input_tokens", 0.0)) for r in parts]),
                "avg_chunk_tokens": mean([float(r.get("chunk_tokens", 0.0)) for r in parts]),
                "avg_output_tokens": mean([float(r.get("output_tokens", 0.0)) for r in parts]),
                "latency_sec_sum": sum(float(r.get("latency_sec", 0.0)) for r in parts),
            }
        )

    bucket_rows = []
    for (model, name), parts in sorted(by_bucket.items()):
        bucket_rows.append(
            {
                "model": model,
                "bucket": name,
                "num_samples": len(parts),
                "exact": mean([float(r.get("exact", 0.0)) for r in parts]),
                "contains": mean([float(r.get("contains", 0.0)) for r in parts]),
            }
        )

    write_csv(results_dir / "summary.csv", summary_rows)
    write_md(results_dir / "summary.md", summary_rows)
    write_csv(results_dir / "position_buckets.csv", bucket_rows)
    write_md(results_dir / "position_buckets.md", bucket_rows)
    print((results_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

