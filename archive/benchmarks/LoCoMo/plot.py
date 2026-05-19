#!/usr/bin/env python
"""Plot LoCoMo aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="full")
    return parser.parse_args()


def _save(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight", dpi=200)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")


def _svg_bar_chart(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 760, 420
    left, top, chart_w, chart_h = 70, 55, 650, 280
    n = max(len(labels), 1)
    bar_w = chart_w / n * 0.62
    gap = chart_w / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:13px} .title{font-size:20px;font-weight:600}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>',
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>',
    ]
    for i in range(6):
        y = top + chart_h - chart_h * i / 5
        parts.append(f'<line x1="{left-4}" y1="{y}" x2="{left+chart_w}" y2="{y}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end">{i/5:.1f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        h = chart_h * max(0.0, min(1.0, float(value)))
        y = top + chart_h - h
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="#4C78A8"/>')
        parts.append(f'<text x="{x + bar_w/2}" y="{y-6}" text-anchor="middle">{value:.3f}</text>')
        parts.append(
            f'<text x="{x + bar_w/2}" y="{top+chart_h+38}" text-anchor="middle" '
            f'transform="rotate(20 {x + bar_w/2},{top+chart_h+38})">{label}</text>'
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_grouped_bar_chart(path: Path, scores: list[dict]) -> None:
    categories = [f"cat{i}" for i in range(1, 6)]
    width, height = 880, 460
    left, top, chart_w, chart_h = 70, 55, 720, 300
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    group_w = chart_w / len(categories)
    bar_w = group_w * 0.75 / max(len(scores), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:13px} .title{font-size:20px;font-weight:600}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">Category-wise LoCoMo QA</text>',
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>',
    ]
    for i in range(6):
        y = top + chart_h - chart_h * i / 5
        parts.append(f'<line x1="{left-4}" y1="{y}" x2="{left+chart_w}" y2="{y}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end">{i/5:.1f}</text>')
    for cidx, cat in enumerate(categories):
        group_x = left + cidx * group_w
        for midx, row in enumerate(scores):
            value = float(row.get(cat, 0.0))
            h = chart_h * max(0.0, min(1.0, value))
            x = group_x + group_w * 0.12 + midx * bar_w
            y = top + chart_h - h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[midx % len(colors)]}"/>')
        parts.append(f'<text x="{group_x + group_w/2}" y="{top+chart_h+24}" text-anchor="middle">{cat}</text>')
    legend_x = left + chart_w + 15
    for midx, row in enumerate(scores):
        y = top + midx * 22
        parts.append(f'<rect x="{legend_x}" y="{y-12}" width="14" height="14" fill="{colors[midx % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{y}">{row["model"]}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_heatmap(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    samples = sorted({r["sample_id"] for r in rows})
    values = {(r["sample_id"], r["model"]): float(r["score"]) for r in rows}
    cell_w, cell_h = 120, 30
    left, top = 130, 50
    width = left + cell_w * len(models) + 40
    height = top + cell_h * len(samples) + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:13px} .title{font-size:20px;font-weight:600}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">Per-conversation score</text>',
    ]
    for j, model in enumerate(models):
        parts.append(f'<text x="{left+j*cell_w+cell_w/2}" y="{top-12}" text-anchor="middle">{model}</text>')
    for i, sample in enumerate(samples):
        parts.append(f'<text x="{left-8}" y="{top+i*cell_h+20}" text-anchor="end">{sample}</text>')
        for j, model in enumerate(models):
            value = values.get((sample, model), 0.0)
            green = int(245 - value * 120)
            color = f"rgb({green},{int(245 - value * 40)},{green})"
            x, y = left + j * cell_w, top + i * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#fff"/>')
            parts.append(f'<text x="{x+cell_w/2}" y="{y+20}" text-anchor="middle">{value:.3f}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _fallback_svg(run_dir: Path, plot_dir: Path, scores: list[dict]) -> None:
    _svg_bar_chart(
        plot_dir / "overall_scores.svg",
        [r["model"] for r in scores],
        [float(r["overall"]) for r in scores],
        "Overall",
    )
    _svg_grouped_bar_chart(plot_dir / "category_scores.svg", scores)
    conv_path = run_dir / "conversation_scores.csv"
    if conv_path.exists():
        with conv_path.open("r", encoding="utf-8") as f:
            _svg_heatmap(plot_dir / "conversation_heatmap.svg", list(csv.DictReader(f)))
    print(f"matplotlib/pandas not available; wrote SVG plots under {plot_dir}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_name
    plot_dir = run_dir / "plots"
    with (run_dir / "scores.json").open("r", encoding="utf-8") as f:
        scores = json.load(f)

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        _fallback_svg(run_dir, plot_dir, scores)
        return

    df = pd.DataFrame(scores)
    if df.empty:
        raise ValueError("No scores found. Run report.py first.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["model"], df["overall"])
    ax.set_ylabel("LoCoMo QA score")
    ax.set_ylim(0, 1)
    ax.set_title("Overall")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, plot_dir / "overall_scores")
    plt.close(fig)

    cat_cols = [f"cat{i}" for i in range(1, 6)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(cat_cols))
    width = 0.8 / max(len(df), 1)
    for idx, row in df.iterrows():
        ax.bar(
            [v + idx * width for v in x],
            [row[c] for c in cat_cols],
            width=width,
            label=row["model"],
        )
    ax.set_xticks([v + width * (len(df) - 1) / 2 for v in x])
    ax.set_xticklabels(cat_cols)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Category-wise LoCoMo QA")
    _save(fig, plot_dir / "category_scores")
    plt.close(fig)

    conv_path = run_dir / "conversation_scores.csv"
    if conv_path.exists():
        conv = pd.read_csv(conv_path)
        if not conv.empty:
            pivot = conv.pivot(index="sample_id", columns="model", values="score")
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="YlGn")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("Per-conversation score")
            fig.colorbar(im, ax=ax, label="Score")
            _save(fig, plot_dir / "conversation_heatmap")
            plt.close(fig)
    print(f"Wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
