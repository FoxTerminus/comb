#!/usr/bin/env python3
"""Plot LongBench score comparisons from per-run summary files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_mpl_cache = Path("/data3/junhaohu/comb/.cache/matplotlib")
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "2wikimqa": "2WikiMQA",
    "musique": "MuSiQue",
    "multi_news": "MultiNews",
    "samsum": "SAMSum",
}
QA_DATASETS = ["hotpotqa", "2wikimqa", "musique"]
SUMMARY_DATASETS = ["multi_news", "samsum"]

DEFAULT_ORDER = ["llama31_8b", "normal_step_108000", "long_step_49955", "cross_step_27165", "kv_step_27165", "e32_step_27165", "e32_step_30000"]


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.weight"] = "medium"
plt.rcParams["hatch.linewidth"] = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LongBench results from summary JSON files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing one subdirectory per run.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated run subdirectory names. Missing runs are skipped.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save figures. Defaults to RESULTS_DIR/plots.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported summary format in {path}")


def find_run_summary(run_dir: Path, run_name: str) -> Path | None:
    preferred = run_dir / f"{run_name}_summary.json"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("*_summary.json"))
    if candidates:
        return candidates[0]
    return None


def collect_scores(results_dir: Path, run_names: list[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for run_name in run_names:
        run_dir = results_dir / run_name
        if not run_dir.is_dir():
            print(f"warning: missing run directory: {run_dir}")
            continue
        summary_path = find_run_summary(run_dir, run_name)
        if summary_path is None:
            print(f"warning: no summary file found in {run_dir}")
            continue
        run_scores: dict[str, float] = {}
        for item in load_summary(summary_path):
            dataset = item.get("dataset")
            score = item.get("score")
            if dataset is not None and score is not None:
                run_scores[str(dataset)] = float(score)
        if run_scores:
            scores[run_name] = run_scores
    return scores


def percent(value: float) -> str:
    return f"{value * 100:.0f}"


def plot_grouped_bars(
    scores: dict[str, dict[str, float]],
    datasets: list[str],
    ylabel: str,
    title: str,
    output_stem: Path,
) -> None:
    run_names = list(scores)
    if not run_names:
        raise ValueError("No runs to plot.")

    values = np.array([[scores[run].get(dataset, np.nan) for dataset in datasets] for run in run_names])
    x = np.arange(len(datasets))
    total_width = min(0.78, 0.18 * len(run_names) + 0.18)
    width = total_width / len(run_names)
    offsets = (np.arange(len(run_names)) - (len(run_names) - 1) / 2) * width

    colors = ["#BC3D27", "#0271B4", "#DF862B", "#228350", "#7776B1", "#925f36"]
    hatches = ["xxxx", "//", "\\\\", "++", "oo", ".."]

    fig, ax = plt.subplots(figsize=(max(4.2, len(datasets) * 1.35), 2.8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bars = []
    for i, run_name in enumerate(run_names):
        bar = ax.bar(
            x + offsets[i],
            values[i],
            width=width,
            label=run_name,
            edgecolor=colors[i % len(colors)],
            fill=False,
            hatch=hatches[i % len(hatches)],
            linewidth=1.4,
        )
        bars.append(bar)
        for j, value in enumerate(values[i]):
            if np.isnan(value):
                continue
            ax.text(
                x[j] + offsets[i],
                value + 0.012,
                percent(value),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x, [DATASET_LABELS[d] for d in datasets])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}"))
    ax.text(0, 1.04, "%", transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
    ymax = np.nanmax(values) if np.isfinite(values).any() else 1.0
    ax.set_ylim(0.0, max(0.1, ymax * 1.22))
    ax.legend(frameon=False, ncol=min(len(run_names), 3), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_table(scores: dict[str, dict[str, float]], output_path: Path) -> None:
    datasets = QA_DATASETS + SUMMARY_DATASETS
    lines = ["run," + ",".join(datasets)]
    for run_name, run_scores in scores.items():
        values = [f"{run_scores.get(dataset, float('nan')):.8f}" for dataset in datasets]
        lines.append(run_name + "," + ",".join(values))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_names = [name.strip() for name in args.runs.split(",") if name.strip()]
    output_dir = args.output_dir or args.results_dir / "plots"
    scores = collect_scores(args.results_dir, run_names)
    if not scores:
        raise SystemExit("No scores found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(scores, output_dir / "score_table.csv")
    plot_grouped_bars(
        scores,
        QA_DATASETS,
        ylabel="F1 Score (%)",
        title="LongBench QA",
        output_stem=output_dir / "longbench_qa_f1",
    )
    plot_grouped_bars(
        scores,
        SUMMARY_DATASETS,
        ylabel="Rouge-L Score (%)",
        title="LongBench Summarization",
        output_stem=output_dir / "longbench_summary_rougel",
    )
    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
