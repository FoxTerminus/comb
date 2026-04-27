"""Plot Pareto-style figures from Phase 4 sweep analysis tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark sweep analysis")
    parser.add_argument("--overall-csv", default="benchmarks/reports/phase4/overall_by_ratio.csv")
    parser.add_argument("--output-dir", default="benchmarks/reports/phase4")
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, float]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "compression_ratio": float(row["compression_ratio"]),
                    "token_f1": float(row["token_f1"]),
                    "max_peak_memory_gb": float(row["max_peak_memory_gb"]),
                    "avg_prefill_latency_s": float(row["avg_prefill_latency_s"]),
                    "avg_decode_latency_s": float(row["avg_decode_latency_s"]),
                    "avg_chunk_tokens": float(row["avg_chunk_tokens"]),
                }
            )
    return sorted(rows, key=lambda item: item["compression_ratio"], reverse=True)


def annotate_points(rows: list[dict[str, float]]) -> None:
    for row in rows:
        plt.annotate(
            str(row["compression_ratio"]),
            (row["x"], row["y"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )


def plot_metric(rows: list[dict[str, float]], x_key: str, y_key: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    plot_rows = [{"x": row[x_key], "y": row[y_key], "compression_ratio": row["compression_ratio"]} for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot([row["x"] for row in plot_rows], [row["y"] for row in plot_rows], marker="o")
    annotate_points(plot_rows)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.overall_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        rows,
        "max_peak_memory_gb",
        "token_f1",
        "Max Peak Memory (GB)",
        "Token F1",
        output_dir / "pareto_memory_vs_f1.png",
    )
    plot_metric(
        rows,
        "avg_decode_latency_s",
        "token_f1",
        "Average Decode Latency (s)",
        "Token F1",
        output_dir / "pareto_decode_latency_vs_f1.png",
    )
    plot_metric(
        rows,
        "avg_chunk_tokens",
        "max_peak_memory_gb",
        "Average Chunk Tokens",
        "Max Peak Memory (GB)",
        output_dir / "chunk_tokens_vs_memory.png",
    )
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
