"""Summarize all JSONL files listed by a sweep manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.metrics import contains_match, exact_match, token_f1
from benchmarks.scripts.schema import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a compression sweep manifest")
    parser.add_argument("manifest_json")
    parser.add_argument("--output-csv", default="benchmarks/reports/phase3_sweep_summary.csv")
    return parser.parse_args()


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        rows.extend(read_jsonl(item["path"]))

    groups: dict[tuple[float, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(float(row.get("compression_ratio", 1.0)), row["benchmark"], row["task"])].append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "compression_ratio",
                "benchmark",
                "task",
                "num_examples",
                "exact_match",
                "contains_match",
                "token_f1",
                "avg_prompt_tokens",
                "avg_chunk_tokens",
                "avg_decoder_tokens",
                "avg_generated_tokens",
                "avg_peak_memory_gb",
                "avg_prefill_latency_s",
                "avg_decode_latency_s",
                "avg_tokens_per_second",
                "num_errors",
            ],
        )
        writer.writeheader()
        for (ratio, benchmark, task), items in sorted(groups.items()):
            answers = [item for item in items if item.get("answer") is not None]
            writer.writerow(
                {
                    "compression_ratio": ratio,
                    "benchmark": benchmark,
                    "task": task,
                    "num_examples": len(items),
                    "exact_match": mean([float(exact_match(i.get("prediction"), i.get("answer"))) for i in answers]),
                    "contains_match": mean([float(contains_match(i.get("prediction"), i.get("answer"))) for i in answers]),
                    "token_f1": mean([token_f1(i.get("prediction"), i.get("answer")) or 0.0 for i in answers]),
                    "avg_prompt_tokens": mean([float(i["prompt_tokens"]) for i in items]),
                    "avg_chunk_tokens": mean([float(i["chunk_tokens"]) for i in items]),
                    "avg_decoder_tokens": mean([float(i["decoder_tokens"]) for i in items]),
                    "avg_generated_tokens": mean([float(i["generated_tokens"]) for i in items]),
                    "avg_peak_memory_gb": mean([float(i["peak_memory_gb"]) for i in items if i.get("peak_memory_gb") is not None]),
                    "avg_prefill_latency_s": mean([float(i["prefill_latency_s"]) for i in items]),
                    "avg_decode_latency_s": mean([float(i["decode_latency_s"]) for i in items]),
                    "avg_tokens_per_second": mean(
                        [float(i["tokens_per_second"]) for i in items if i.get("tokens_per_second") is not None]
                    ),
                    "num_errors": sum(1 for i in items if i.get("error")),
                }
            )
    print(f"Wrote sweep summary to {output_path}")


if __name__ == "__main__":
    main()
