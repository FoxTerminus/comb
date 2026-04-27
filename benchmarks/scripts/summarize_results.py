"""Summarize benchmark JSONL records into a compact CSV table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.schema import read_jsonl
from benchmarks.scripts.metrics import contains_match, exact_match, token_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSONL results")
    parser.add_argument("input_jsonl")
    parser.add_argument("--output-csv", default="benchmarks/reports/smoke_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["task"], float(row.get("compression_ratio", 1.0)))].append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "task",
                "compression_ratio",
                "num_examples",
                "exact_match",
                "contains_match",
                "token_f1",
                "avg_peak_memory_gb",
                "avg_prefill_latency_s",
                "avg_decode_latency_s",
                "avg_tokens_per_second",
                "num_errors",
            ],
        )
        writer.writeheader()
        for (benchmark, task, compression_ratio), items in sorted(groups.items()):
            em_values = [
                exact_match(item.get("prediction"), item.get("answer"))
                for item in items
                if item.get("answer") is not None
            ]
            contains_values = [
                contains_match(item.get("prediction"), item.get("answer"))
                for item in items
                if item.get("answer") is not None
            ]
            f1_values = [
                token_f1(item.get("prediction"), item.get("answer"))
                for item in items
                if item.get("answer") is not None
            ]
            memory_values = [item["peak_memory_gb"] for item in items if item["peak_memory_gb"] is not None]
            tps_values = [item["tokens_per_second"] for item in items if item["tokens_per_second"] is not None]
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "task": task,
                    "compression_ratio": compression_ratio,
                    "num_examples": len(items),
                    "exact_match": sum(em_values) / len(em_values) if em_values else "",
                    "contains_match": sum(contains_values) / len(contains_values) if contains_values else "",
                    "token_f1": sum(f1_values) / len(f1_values) if f1_values else "",
                    "avg_peak_memory_gb": sum(memory_values) / len(memory_values) if memory_values else "",
                    "avg_prefill_latency_s": sum(item["prefill_latency_s"] for item in items) / len(items),
                    "avg_decode_latency_s": sum(item["decode_latency_s"] for item in items) / len(items),
                    "avg_tokens_per_second": sum(tps_values) / len(tps_values) if tps_values else "",
                    "num_errors": sum(1 for item in items if item.get("error")),
                }
            )
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
