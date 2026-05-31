#!/usr/bin/env python3
"""Plot training and validation loss from a CombLlama training output dir."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = "/data3/junhaohu/checkpoints/CombLlama_e32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help="Training output directory containing training_log.csv and validation_log.csv.",
    )
    parser.add_argument("--output", default=None, help="Output image path. Defaults to <output_dir>/loss_plot.png.")
    parser.add_argument("--smooth-window", type=int, default=10, help="Moving-average window over logged train points.")
    parser.add_argument("--no-instant", action="store_true", help="Do not draw instant_loss when present.")
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        smoothed.append(running_sum / count)
    return smoothed


def read_training_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing training log: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_validation_log(path: Path) -> dict[str, dict[str, list[float]]]:
    by_dataset: dict[str, dict[str, list[float]]] = {}
    if not path.exists():
        return by_dataset
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("loss"):
                continue
            dataset = row["dataset"]
            by_dataset.setdefault(dataset, {"steps": [], "losses": []})
            by_dataset[dataset]["steps"].append(int(row["step"]))
            by_dataset[dataset]["losses"].append(float(row["loss"]))
    return by_dataset


def plot_loss(output_dir: Path, output_path: Path, smooth_window: int, draw_instant: bool) -> None:
    train_rows = read_training_log(output_dir / "training_log.csv")
    val_by_dataset = read_validation_log(output_dir / "validation_log.csv")

    steps = [int(row["step"]) for row in train_rows]
    train_loss = [float(row["loss"]) for row in train_rows]
    instant_loss = [float(row["instant_loss"]) for row in train_rows if row.get("instant_loss")]
    datasets = [row.get("dataset", "") for row in train_rows]
    smoothed = moving_average(train_loss, smooth_window)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        dpi=170,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    ax = axes[0]
    ax.plot(steps, train_loss, color="#1f77b4", linewidth=1.2, label="training loss")
    if smooth_window > 1:
        ax.plot(steps, smoothed, color="#d62728", linewidth=2.0, label=f"moving avg ({smooth_window} logs)")
    if draw_instant and len(instant_loss) == len(steps):
        ax.plot(steps, instant_loss, color="#9ecae1", linewidth=0.7, alpha=0.45, label="instant loss")

    last_dataset = None
    y_top = max(train_loss) if train_loss else 1.0
    for step, dataset in zip(steps, datasets):
        if dataset != last_dataset:
            if last_dataset is not None:
                ax.axvline(step, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45)
            if dataset:
                ax.text(step, y_top * 0.98, dataset, rotation=90, va="top", ha="right", fontsize=8, color="#444444")
            last_dataset = dataset

    ax.set_title("CombLlama Training Loss")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    ax_val = axes[1]
    colors = {
        "__mean__": "#d62728",
        "SQuAD": "#2ca02c",
        "Natural-Instructions": "#9467bd",
        "XSum": "#ff7f0e",
        "Super-Natural-Instructions": "#8c564b",
    }
    for dataset, data in val_by_dataset.items():
        ax_val.plot(
            data["steps"],
            data["losses"],
            marker="o" if dataset == "__mean__" else "x",
            linewidth=2.0 if dataset == "__mean__" else 1.2,
            markersize=4,
            label=dataset,
            color=colors.get(dataset),
        )
    ax_val.set_title("Validation Loss")
    ax_val.set_xlabel("Optimizer step")
    ax_val.set_ylabel("Loss")
    ax_val.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    if val_by_dataset:
        ax_val.legend(frameon=False, fontsize=8, ncol=2)
    else:
        ax_val.text(0.5, 0.5, "No validation points yet", transform=ax_val.transAxes, ha="center", va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else output_dir / "loss_plot.png"
    plot_loss(output_dir, output_path, args.smooth_window, not args.no_instant)


if __name__ == "__main__":
    main()
