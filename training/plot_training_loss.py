#!/usr/bin/env python3
"""Plot a training-loss CSV produced by training/train.py."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comb-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to training_loss.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image (default: <csv directory>/training_loss_plot.png)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="Moving-average windows in optimizer steps (default: 100 1000)",
    )
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default="Training loss")
    parser.add_argument(
        "--hide-raw", action="store_true", help="Do not draw individual-step loss"
    )
    return parser.parse_args()


def load_rows(path: Path, min_step: int | None, max_step: int | None):
    rows: list[tuple[int, float, str, str]] = []
    skipped = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"optimizer_step", "loss"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        for record in reader:
            try:
                step = int(record["optimizer_step"])
                loss = float(record["loss"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if not math.isfinite(loss):
                skipped += 1
                continue
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            rows.append(
                (step, loss, record.get("dataset", ""), record.get("bucket", ""))
            )
    if not rows:
        raise ValueError(f"No finite loss rows selected from {path}")
    rows.sort(key=lambda item: item[0])
    return rows, skipped


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("moving-average windows must be positive")
    if len(values) < window:
        return np.empty(0, dtype=np.float64)
    cumulative = np.cumsum(np.insert(values, 0, 0.0), dtype=np.float64)
    return (cumulative[window:] - cumulative[:-window]) / window


def transitions(rows: list[tuple[int, float, str, str]]):
    result: list[tuple[int, str]] = []
    previous: tuple[str, str] | None = None
    for step, _, dataset, bucket in rows:
        state = (dataset, bucket)
        if state != previous:
            dataset_label = dataset or "dataset?"
            bucket_label = f"bucket {bucket}" if bucket != "" else ""
            result.append(
                (step, " / ".join(x for x in (dataset_label, bucket_label) if x))
            )
            previous = state
    return result


def main() -> None:
    args = parse_args()
    rows, skipped = load_rows(args.csv, args.min_step, args.max_step)
    steps = np.asarray([row[0] for row in rows], dtype=np.int64)
    losses = np.asarray([row[1] for row in rows], dtype=np.float64)
    output = args.output or args.csv.with_name("training_loss_plot.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(14, 7))
    if not args.hide_raw:
        axis.plot(
            steps,
            losses,
            color="#9aa0a6",
            linewidth=0.35,
            alpha=0.18,
            rasterized=True,
            label="per-step loss",
        )

    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for index, window in enumerate(args.windows):
        average = moving_average(losses, window)
        if len(average):
            axis.plot(
                steps[window - 1 :],
                average,
                color=colors[index % len(colors)],
                linewidth=1.4 if window < 1000 else 2.0,
                label=f"{window}-step mean",
            )

    for index, (step, label) in enumerate(transitions(rows)):
        axis.axvline(step, color="#555555", linewidth=0.7, alpha=0.45)
        axis.annotate(
            label,
            xy=(step, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4 - 13 * (index % 2)),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=7,
            color="#444444",
        )

    axis.set_title(args.title)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper right")
    axis.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Wrote {output} from {len(rows)} rows "
        f"(steps {steps[0]}-{steps[-1]}, skipped {skipped})."
    )


if __name__ == "__main__":
    main()
