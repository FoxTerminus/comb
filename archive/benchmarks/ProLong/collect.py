#!/usr/bin/env python
"""Collect ProLong validation summaries into a table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data3/junhaohu/comb/benchmarks/ProLong/results/validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    shard_rows = []
    for path in sorted(results_dir.glob("*_summary.json")):
        shard_rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not shard_rows:
        raise FileNotFoundError(f"No *_summary.json files under {results_dir}")

    grouped = defaultdict(list)
    for row in shard_rows:
        grouped[row["model"]].append(row)

    rows = []
    for model, parts in sorted(grouped.items()):
        tokens = sum(int(p.get("tokens", 0)) for p in parts)
        weighted_loss = sum(float(p.get("loss", 0.0)) * int(p.get("tokens", 0)) for p in parts)
        loss = weighted_loss / max(tokens, 1)
        rows.append({
            "model": model,
            "num_shards": len(parts),
            "num_samples": sum(int(p.get("num_samples", 0)) for p in parts),
            "tokens": tokens,
            "loss": loss,
            "ppl": math.exp(min(loss, 20.0)),
            "elapsed_sec_sum": sum(float(p.get("elapsed_sec", 0.0)) for p in parts),
            "elapsed_sec_max": max(float(p.get("elapsed_sec", 0.0)) for p in parts),
            "data_dir": parts[0].get("data_dir"),
            "block_size": parts[0].get("block_size"),
        })

    keys = [
        "model", "num_shards", "num_samples", "tokens", "loss", "ppl",
        "elapsed_sec_sum", "elapsed_sec_max", "data_dir", "block_size",
    ]
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})

    md_path = results_dir / "summary.md"
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in rows:
        vals = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                value = round(value, 6)
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
