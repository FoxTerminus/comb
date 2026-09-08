#!/usr/bin/env python3
"""Refresh the active curriculum loss plot at fixed optimizer-step intervals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        type=Path,
        default=Path(
            "/data3/junhaohu/checkpoints/CombLlamaFrozen32/watchdog_status.json"
        ),
        help="Frozen32 watchdog status JSON",
    )
    parser.add_argument(
        "--interval", type=int, default=5000, help="Optimizer-step interval"
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_status(path: Path) -> dict:
    return json.loads(path.read_text())


def render(plot_script: Path, output_dir: Path, step: int, stage: str) -> Path:
    csv_path = output_dir / "training_loss.csv"
    final_path = output_dir / "training_loss_plot.png"
    temporary = output_dir / ".training_loss_plot.tmp.png"
    command = [
        sys.executable,
        str(plot_script),
        str(csv_path),
        "--output",
        str(temporary),
        "--max-step",
        str(step),
        "--title",
        f"Frozen32 PIC - {stage}",
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/comb-matplotlib-cache")
    subprocess.run(command, check=True, env=environment)
    os.replace(temporary, final_path)
    return final_path


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    plot_script = Path(__file__).with_name("plot_training_loss.py")
    active_output: Path | None = None
    last_milestone: int | None = None

    while True:
        try:
            status = read_status(args.status)
            step = int(status["current_step"])
            stage = str(status["stage"])
            output_dir = Path(status["output_dir"])
            milestone = step // args.interval
            stage_changed = output_dir != active_output
            should_render = stage_changed or last_milestone is None or milestone > last_milestone
            if should_render and (output_dir / "training_loss.csv").exists():
                path = render(plot_script, output_dir, step, stage)
                print(
                    f"updated {path} at step {step} "
                    f"(milestone {milestone * args.interval})",
                    flush=True,
                )
                active_output = output_dir
                last_milestone = milestone
            if args.once:
                return
            if stage == "Super-NI" and step >= int(status["expected_final_step"]):
                return
        except Exception as error:
            print(f"plot watcher error: {error!r}", flush=True)
            if args.once:
                raise
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
