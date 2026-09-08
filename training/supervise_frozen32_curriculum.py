#!/usr/bin/env python3
"""Persistently validate and advance the Frozen32 PIC data curriculum."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

from training.checkpoint_integrity import valid_torch_zip


ROOT = Path("/data3/junhaohu/checkpoints/CombLlamaFrozen32")
TRAINING = Path("/data3/junhaohu/comb/training")
LAUNCHERS = TRAINING.parent / "scripts/training"
STATUS = ROOT / "curriculum_supervisor_status.json"
POLL_SECONDS = 60

STAGES = [
    {
        "name": "SQuAD",
        "output": ROOT / "squad_stage1",
        "final_step": 4073,
        "launch": None,
    },
    {
        "name": "Natural-Instructions",
        "output": ROOT / "ni_stage2",
        "final_step": 196690,
        "launch": LAUNCHERS / "launch_frozen32_ni.sh",
    },
    {
        "name": "XSum",
        "output": ROOT / "xsum_stage3",
        "final_step": 203068,
        "launch": LAUNCHERS / "launch_frozen32_xsum.sh",
    },
    {
        "name": "Super-Natural-Instructions",
        "output": ROOT / "superni_stage4",
        "final_step": 265285,
        "launch": LAUNCHERS / "launch_frozen32_superni.sh",
    },
]


def write_status(**fields) -> None:
    payload = {"updated_unix": time.time(), **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, STATUS)


def last_loss_step(output: Path) -> int | None:
    path = output / "training_loss.csv"
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return int(rows[-1]["optimizer_step"])


def process_running(output: Path) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "train_llama_frozen32.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    needle = f"--output-dir {output}"
    return any(needle in line for line in result.stdout.splitlines())


def wait_for_external_stage(stage: dict) -> None:
    output = stage["output"]
    final_step = stage["final_step"]
    final_eval = output / f"context_dependency_step_{final_step:08d}.json"
    while True:
        current = last_loss_step(output)
        if final_eval.exists():
            return
        running = process_running(output)
        write_status(
            state="waiting",
            stage=stage["name"],
            current_step=current,
            expected_final_step=final_step,
            process_running=running,
        )
        if not running:
            raise RuntimeError(
                f"{stage['name']} stopped at {current}, expected {final_step}"
            )
        time.sleep(POLL_SECONDS)


def validate_stage(stage: dict) -> dict:
    output = stage["output"]
    final_step = stage["final_step"]
    loss_path = output / "training_loss.csv"
    with loss_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["optimizer_step"]) != final_step:
        raise RuntimeError(f"{stage['name']} final loss row is incomplete")
    losses = [float(row["loss"]) for row in rows]
    if any(not math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"{stage['name']} contains non-finite loss")

    checkpoint = output / f"step_{final_step:08d}"
    shards = sorted(checkpoint.glob("mp_rank_*_model_states.pt"))
    if len(shards) != 4 or not all(valid_torch_zip(path) for path in shards):
        raise RuntimeError(f"{stage['name']} final TP checkpoint is incomplete")

    eval_path = output / f"context_dependency_step_{final_step:08d}.json"
    metrics = json.loads(eval_path.read_text())
    required_finite = (
        "correct_context_nll",
        "context_nll_gap",
        "distinct_context_nll_gap",
        "no_context_nll_gap",
        "positive_batch_fraction",
    )
    if any(not math.isfinite(float(metrics[key])) for key in required_finite):
        raise RuntimeError(f"{stage['name']} evaluation contains non-finite metrics")
    if float(metrics["context_nll_gap"]) <= 0.0:
        raise RuntimeError(f"{stage['name']} no longer distinguishes shuffled context")
    if float(metrics["distinct_context_nll_gap"]) <= 0.0:
        raise RuntimeError(f"{stage['name']} no longer distinguishes distinct context")
    if float(metrics["no_context_nll_gap"]) <= 0.1:
        raise RuntimeError(f"{stage['name']} context-use margin is too small")
    if float(metrics["positive_batch_fraction"]) < 0.75:
        raise RuntimeError(f"{stage['name']} context preference is not consistent")

    tail = losses[-min(500, len(losses)):]
    return {
        "final_step": final_step,
        "final_loss": losses[-1],
        "tail_loss_mean": sum(tail) / len(tail),
        "checkpoint_bytes": sum(path.stat().st_size for path in shards),
        "correct_context_nll": metrics["correct_context_nll"],
        "context_nll_gap": metrics["context_nll_gap"],
        "distinct_context_nll_gap": metrics["distinct_context_nll_gap"],
        "no_context_nll_gap": metrics["no_context_nll_gap"],
        "positive_batch_fraction": metrics["positive_batch_fraction"],
    }


def wait_for_gpus() -> None:
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        used = [int(value.strip()) for value in result.stdout.splitlines()]
        if len(used) >= 4 and all(value < 5000 for value in used[:4]):
            return
        write_status(state="waiting_for_gpus", gpu_memory_mib=used[:4])
        time.sleep(POLL_SECONDS)


def run_stage(stage: dict) -> None:
    wait_for_gpus()
    output = stage["output"]
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_stdout.log"
    write_status(
        state="launching",
        stage=stage["name"],
        expected_final_step=stage["final_step"],
        command=str(stage["launch"]),
    )
    with log_path.open("a") as log:
        result = subprocess.run(
            ["bash", str(stage["launch"])],
            cwd=TRAINING,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{stage['name']} launcher exited {result.returncode}")


def main() -> None:
    completed = {}
    try:
        for index, stage in enumerate(STAGES):
            final_eval = stage["output"] / (
                f"context_dependency_step_{stage['final_step']:08d}.json"
            )
            if not final_eval.exists():
                if stage["launch"] is None:
                    wait_for_external_stage(stage)
                else:
                    run_stage(stage)
            completed[stage["name"]] = validate_stage(stage)
            write_status(
                state="stage_validated",
                stage=stage["name"],
                completed=completed,
                remaining=[item["name"] for item in STAGES[index + 1 :]],
            )
        write_status(state="complete", completed=completed)
    except Exception as error:
        write_status(state="failed", error=repr(error), completed=completed)
        raise


if __name__ == "__main__":
    main()
