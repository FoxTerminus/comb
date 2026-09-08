#!/usr/bin/env python3
"""Restart a pure true-TP Comb run from its latest committed checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from training.supervise_single_training import (
    append_event,
    committed_checkpoint,
    quarantine_uncommitted_checkpoints,
    rollback_loss_log,
    rotate_stdout,
    training_processes,
    validate_restart_data,
)


REPO = Path("/data3/junhaohu/comb")
LAUNCHER = REPO / "training/run_full_resume_true_tp.sh"
FINAL_STEP = 265_287
SOURCE_FILES = (
    "training/train_llama_repro.py",
    "training/train_llama_true_tp_repro.py",
    "training/true_tp_comb_adapter.py",
    "training/ds_llama_config.json",
    "training/ds_llama_true_tp_stage0_config.json",
    "training/launch_llama_repro_true_tp.sh",
    "training/run_full_resume_true_tp.sh",
    "comb/integration/hf/CombLlama.py",
    "data/base.py",
)


def source_hashes() -> dict[str, str]:
    return {
        relative: hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def validate_or_create_source_lock(root: Path) -> dict[str, str]:
    path = root / "true_tp_supervisor_source_lock.json"
    current = source_hashes()
    if path.exists():
        expected = json.loads(path.read_text())["source_sha256"]
        if current != expected:
            changed = sorted(name for name in current if current[name] != expected.get(name))
            raise RuntimeError(f"unreviewed true-TP restart source: {changed}")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"source_sha256": current}, indent=2) + "\n")
        temporary.replace(path)
    return current


def validate_universal_baseline(root: Path, tag: str, step: int) -> None:
    checkpoint = root / tag
    marker = json.loads(
        (checkpoint / "comb_universal_conversion_complete.json").read_text()
    )
    if (
        tag != f"step_{step:08d}"
        or int(marker.get("file_count", 0)) <= 0
        or int(marker.get("total_bytes", 0)) <= 0
        or not (checkpoint / "mp_rank_00_model_states.pt").is_file()
    ):
        raise RuntimeError("invalid universal baseline checkpoint")


def validate_baseline_if_needed(
    root: Path, baseline_root: Path, baseline_tag: str, baseline_step: int
) -> None:
    """Require the universal baseline only when no native resume exists."""
    if not (root / "latest_repro.json").is_file():
        validate_universal_baseline(baseline_root, baseline_tag, baseline_step)


def final_context_complete(root: Path) -> bool:
    path = root / f"context_dependency_step_{FINAL_STEP:08d}.json"
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return (
        int(payload.get("optimizer_step", -1)) == FINAL_STEP
        and int(payload.get("examples", 0)) == 64
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-tag", required=True)
    parser.add_argument("--baseline-step", type=int, required=True)
    parser.add_argument(
        "--ds-config", default="ds_llama_true_tp_stage0_config.json"
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--confirmations", type=int, default=3)
    parser.add_argument("--minimum-free-gib", type=int, default=300)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_root = args.baseline_root.resolve()
    if "/" in args.ds_config or not (REPO / "training" / args.ds_config).is_file():
        raise ValueError("--ds-config must name a config file in training/")
    root.mkdir(parents=True, exist_ok=True)
    stop_file = args.stop_file or root / "STOP_AUTORESTART"
    if args.tp_size <= 1 or len(args.gpu_ids.split(",")) != args.tp_size:
        raise ValueError("pure TP requires one distinct GPU ID per TP rank")
    if len(set(args.gpu_ids.split(","))) != args.tp_size:
        raise ValueError("GPU IDs must be distinct")
    validate_baseline_if_needed(
        root, baseline_root, args.baseline_tag, args.baseline_step
    )
    hashes = validate_or_create_source_lock(root)
    datasets = validate_restart_data()

    lock = (root / "true_tp_training_supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another true-TP supervisor is active") from error

    absent = 0
    while True:
        if stop_file.exists():
            append_event(root, {"time": time.time(), "event": "stop_file_seen"})
            return
        pids = training_processes(root)
        if pids:
            absent = 0
            if args.once:
                print(json.dumps({"training_pids": pids, "action": "none"}))
                return
            time.sleep(args.poll_seconds)
            continue
        absent += 1
        if absent < args.confirmations:
            if args.once:
                print(json.dumps({"training_pids": [], "action": "await_confirmation"}))
                return
            time.sleep(args.poll_seconds)
            continue

        if (root / "latest_repro.json").is_file():
            tag, step = committed_checkpoint(root, args.tp_size)
            resume_root = root
        else:
            tag, step = args.baseline_tag, args.baseline_step
            resume_root = baseline_root
        if step >= FINAL_STEP and final_context_complete(root):
            append_event(root, {"time": time.time(), "event": "training_complete", "tag": tag})
            return
        free_gib = shutil.disk_usage(root).free / 1024**3
        if free_gib < args.minimum_free_gib:
            append_event(
                root,
                {"time": time.time(), "event": "restart_refused_low_disk", "free_gib": free_gib},
            )
            time.sleep(max(args.poll_seconds, 300))
            continue

        validate_or_create_source_lock(root)
        recovery_id = time.strftime("%Y%m%dT%H%M%S%z")
        quarantined = quarantine_uncommitted_checkpoints(root, step, recovery_id)
        archive = None
        if (root / "training_loss.csv").is_file():
            archive = rollback_loss_log(
                root,
                step,
                recovery_id,
                expected_first_step=args.baseline_step + 1,
            )
        prior_stdout = rotate_stdout(root, recovery_id)
        event = {
            "time": time.time(),
            "event": "restart",
            "resume_root": str(resume_root),
            "resume_tag": tag,
            "discarded_loss_archive": str(archive) if archive else None,
            "quarantined_uncommitted_checkpoints": [str(path) for path in quarantined],
            "prior_stdout": str(prior_stdout) if prior_stdout else None,
            "free_gib": free_gib,
            "source_sha256": hashes,
            "datasets": datasets,
        }
        append_event(root, event)
        environment = os.environ.copy()
        environment.update(
            {
                "COMB_GPU_IDS": args.gpu_ids,
                "COMB_TP_SIZE": str(args.tp_size),
                "COMB_OUTPUT_DIR": str(root),
                "COMB_RESUME_ROOT": str(resume_root),
                "COMB_RESUME_TAG": tag,
                "COMB_DS_CONFIG": args.ds_config,
            }
        )
        result = subprocess.run([str(LAUNCHER)], env=environment, check=False)
        append_event(
            root,
            {"time": time.time(), "event": "training_process_returned", "returncode": result.returncode},
        )
        absent = 0
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
