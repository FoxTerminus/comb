#!/usr/bin/env python
"""Plot Phonebook benchmark summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data3/junhaohu/comb/benchmarks/Phonebook/results/phonebook_32k")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary = read_csv(results_dir / "summary.csv")
    buckets = read_csv(results_dir / "position_buckets.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [row["model"] for row in summary]
    exact = [float(row["exact"]) for row in summary]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(models, exact, color=["#dc2626", "#2563eb", "#16a34a", "#7c3aed"][: len(models)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact match")
    ax.set_title("Phonebook Retrieval Accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "accuracy.png")
    fig.savefig(results_dir / "accuracy.pdf")

    bucket_names = ["early", "middle", "late"]
    values = {(row["model"], row["bucket"]): float(row["exact"]) for row in buckets}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    width = 0.8 / max(len(models), 1)
    xs = list(range(len(bucket_names)))
    for idx, model in enumerate(models):
        offsets = [x - 0.4 + width / 2 + idx * width for x in xs]
        ax.bar(offsets, [values.get((model, b), 0.0) for b in bucket_names], width=width, label=model)
    ax.set_xticks(xs, bucket_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact match")
    ax.set_title("Phonebook Accuracy by Target Position")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "position_buckets.png")
    fig.savefig(results_dir / "position_buckets.pdf")
    print(f"Wrote plots under {results_dir}")


if __name__ == "__main__":
    main()
