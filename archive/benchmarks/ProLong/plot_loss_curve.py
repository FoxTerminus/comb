#!/usr/bin/env python
"""Plot per-sample ProLong validation loss curves from shard details."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data3/junhaohu/comb/benchmarks/ProLong/results/validation")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to validation_loss_curve.png.")
    parser.add_argument("--rolling-window", type=int, default=16)
    return parser.parse_args()


def read_rows(results_dir: Path) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in sorted(results_dir.glob("*_details.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                sample_index = int(row["sample_index"])
                shard_id = int(row["shard_id"])
                num_shards = int(row["num_shards"])
                grouped[model].append(
                    {
                        "global_index": sample_index * num_shards + shard_id,
                        "sample_index": sample_index,
                        "shard_id": shard_id,
                        "loss": float(row["loss"]),
                        "ppl": float(row["ppl"]),
                        "tokens": int(row["tokens"]),
                    }
                )
    if not grouped:
        raise FileNotFoundError(f"No *_details.csv files found under {results_dir}")
    for rows in grouped.values():
        rows.sort(key=lambda x: x["global_index"])
    return grouped


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    output = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        output.append(running / len(queue))
    return output


def write_combined_csv(grouped: dict[str, list[dict[str, float]]], output: Path) -> None:
    rows = []
    for model, model_rows in grouped.items():
        for row in model_rows:
            rows.append({"model": model, **row})
    rows.sort(key=lambda x: (x["model"], x["global_index"]))
    keys = ["model", "global_index", "sample_index", "shard_id", "loss", "ppl", "tokens"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "validation_loss_curve.png"
    combined_csv = output.with_suffix(".csv")
    grouped = read_rows(results_dir)
    write_combined_csv(grouped, combined_csv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "sambay": "#2563eb",
        "sambayoco": "#16a34a",
        "comb-qwen": "#dc2626",
    }

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    for model in sorted(grouped):
        rows = grouped[model]
        xs = [int(row["global_index"]) for row in rows]
        losses = [float(row["loss"]) for row in rows]
        color = colors.get(model)
        ax.plot(xs, losses, color=color, alpha=0.18, linewidth=1.0)
        ax.plot(
            xs,
            rolling_mean(losses, args.rolling_window),
            color=color,
            linewidth=2.0,
            label=f"{model} (rolling {args.rolling_window})",
        )

    ax.set_title("ProLong Validation Loss Curve")
    ax.set_xlabel("Validation sample index")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")
    print(f"Wrote {combined_csv}")


if __name__ == "__main__":
    main()
