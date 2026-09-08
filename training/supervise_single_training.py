#!/usr/bin/env python3
"""Restart the long single-GPU reproduction from its last committed checkpoint."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from training.checkpoint_integrity import valid_torch_zip


ROOT = Path("/data3/junhaohu/checkpoints/Comb_official_reproduction_single_rank")
LAUNCHER = Path("/data3/junhaohu/comb/training/run_full_resume_single.sh")
FINAL_STEP = 265_287
FINAL_CONTEXT_EXAMPLES = 64
REPO = Path("/data3/junhaohu/comb")
RESTART_SOURCE_HASHES = {
    "training/train_llama_repro.py": (
        "4d29e9e523a387c0c60a283080bdafa7bd634484e5e1126f86321aa1ba60deb4"
    ),
    "training/ds_llama_config.json": (
        "7a5effa3e1ddf824a27a9cd415aa6c0b3ee3ba2627973a3d9684f96680796cfc"
    ),
    "comb/integration/hf/CombLlama.py": (
        "bb5be982756197c3dbd6348e89a1ec394dac97e6df690ab151f2cd575131ca27"
    ),
    "data/base.py": (
        "dad1fffc556ca04c17c1115a7dd3b0fdb7a8df6192c4073fdded4e0a03bf81d5"
    ),
}
DATASET_ROOT = Path("/data3/junhaohu/.cache/huggingface/datasets")
RESTART_DATASETS = {
    "SQuAD": (130_319, "be31e3cae4c05432"),
    "Natural-Instructions": (6_164_188, "05fa7b5a243edad9"),
    "XSum": (204_045, "56a2869f9bbf38b0"),
    "Super-Natural-Instructions": (1_990_915, "bd847ff15bcc5ceb"),
}


def training_processes(root: Path) -> list[int]:
    needle = str(root).encode()
    result = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            command = (process / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # DataLoader workers inherit the rank process argv and can briefly
        # outlive a failed rank.  The DeepSpeed launcher is the stable owner
        # of the whole training process tree and remains alive through final
        # evaluation/export.
        if b"deepspeed.launcher.launch" in command and needle in command:
            result.append(int(process.name))
    return result


def committed_checkpoint(root: Path, model_parallel_size: int = 1) -> tuple[str, int]:
    primary_error = None
    try:
        metadata = json.loads((root / "latest_repro.json").read_text())
        return _validate_checkpoint_metadata(
            root, metadata, model_parallel_size, require_latest_pointer=True
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError,
            KeyError, TypeError, ValueError, RuntimeError) as error:
        primary_error = error
    try:
        backup = json.loads((root / "last_verified_checkpoint.json").read_text())
        return _validate_checkpoint_metadata(
            root, backup, model_parallel_size, require_latest_pointer=False
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError,
            KeyError, TypeError, ValueError, RuntimeError) as backup_error:
        raise RuntimeError(
            f"no recoverable committed checkpoint; primary={primary_error!r}; "
            f"backup={backup_error!r}"
        ) from backup_error


def _validate_checkpoint_metadata(
    root: Path,
    metadata: dict,
    model_parallel_size: int,
    *,
    require_latest_pointer: bool,
) -> tuple[str, int]:
    tag = str(metadata["tag"])
    step = int(metadata["optimizer_step"])
    if tag != f"step_{step:08d}":
        raise RuntimeError("checkpoint tag and optimizer step disagree")
    if require_latest_pointer and (root / "latest").read_text().strip() != tag:
        raise RuntimeError("checkpoint pointers disagree")
    checkpoint = root / tag
    expected = tuple(
        checkpoint / f"mp_rank_{rank:02d}_model_states.pt"
        for rank in range(model_parallel_size)
    ) + tuple(
        checkpoint / f"bf16_zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt"
        for rank in range(model_parallel_size)
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
        raise RuntimeError(f"incomplete committed checkpoint: {tag}")
    if any(not valid_torch_zip(path) for path in expected):
        raise RuntimeError(f"invalid checkpoint archive: {tag}")
    recorded_sizes = metadata.get("files")
    if recorded_sizes is not None:
        actual_sizes = {path.name: path.stat().st_size for path in expected}
        if recorded_sizes != actual_sizes:
            raise RuntimeError(f"committed checkpoint sizes changed: {tag}")
    if list(checkpoint.rglob("*.tmp")):
        raise RuntimeError(f"temporary checkpoint files remain: {tag}")
    return tag, step


def rollback_loss_log(
    root: Path,
    checkpoint_step: int,
    recovery_id: str,
    expected_first_step: int = 1,
) -> Path | None:
    path = root / "training_loss.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "optimizer_step" not in fieldnames:
        raise RuntimeError("invalid training loss header")
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != list(range(expected_first_step, expected_first_step + len(steps))):
        raise RuntimeError("training loss is not a contiguous prefix")
    dropped = [row for row in rows if int(row["optimizer_step"]) > checkpoint_step]
    if not dropped:
        return None
    kept = [row for row in rows if int(row["optimizer_step"]) <= checkpoint_step]
    expected_kept = max(checkpoint_step - expected_first_step + 1, 0)
    if len(kept) != expected_kept:
        raise RuntimeError("checkpoint step is outside the logged loss prefix")

    archive = root / f"discarded_loss_after_{checkpoint_step:08d}_{recovery_id}.csv"
    temporary = path.with_suffix(".csv.recovery.tmp")
    if archive.exists() or temporary.exists():
        raise RuntimeError("refusing to overwrite recovery artifacts")
    with archive.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dropped)
        handle.flush()
        os.fsync(handle.fileno())
    with temporary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return archive


def quarantine_uncommitted_checkpoints(
    root: Path, checkpoint_step: int, recovery_id: str
) -> list[Path]:
    quarantined = []
    for path in sorted(root.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]")):
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            continue
        if step <= checkpoint_step:
            continue
        resolved = path.resolve()
        if resolved.parent != root or not resolved.is_dir():
            raise RuntimeError(f"unsafe uncommitted checkpoint path: {resolved}")
        target = root / f"incomplete_{path.name}_{recovery_id}"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite quarantine path: {target}")
        path.rename(target)
        quarantined.append(target)
    return quarantined


def rotate_stdout(root: Path, recovery_id: str) -> Path | None:
    path = Path(str(root) + ".stdout.log")
    if not path.is_file() or path.stat().st_size == 0:
        return None
    target = path.with_name(path.name + f".failed_{recovery_id}")
    if target.exists():
        raise RuntimeError(f"refusing to overwrite prior stdout: {target}")
    path.rename(target)
    return target


def final_artifacts_complete(
    root: Path, *, run_strict_verifiers: bool = True
) -> bool:
    context = root / f"context_dependency_step_{FINAL_STEP:08d}.json"
    export = root / f"hf_step_{FINAL_STEP:08d}"
    try:
        context_payload = json.loads(context.read_text())
        json.loads((export / "config.json").read_text())
        index = json.loads(
            (export / "model.safetensors.index.json").read_text()
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    if (
        int(context_payload.get("optimizer_step", -1)) != FINAL_STEP
        or int(context_payload.get("examples", 0)) != FINAL_CONTEXT_EXAMPLES
    ):
        return False
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shard_names = set(weight_map.values())
    if not all(
        isinstance(name, str)
        and name == Path(name).name
        and name.endswith(".safetensors")
        for name in shard_names
    ):
        return False
    expected_shards = {export / name for name in shard_names}
    existing_shards = set(export.glob("*.safetensors"))
    if existing_shards != expected_shards or any(
        not path.is_file() or path.stat().st_size == 0
        for path in expected_shards
    ):
        return False
    if list(export.rglob("*.tmp")):
        return False
    if not run_strict_verifiers:
        return True
    commands = (
        (
            sys.executable,
            str(REPO / "training/verify_context_dependency.py"),
            str(context),
            "--expected-step",
            str(FINAL_STEP),
            "--expected-examples",
            str(FINAL_CONTEXT_EXAMPLES),
        ),
        (
            sys.executable,
            str(REPO / "training/verify_hf_checkpoint.py"),
            str(export),
            "--expected-parameters",
            "12045391888",
        ),
    )
    verifier_environment = os.environ.copy()
    verifier_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRITON_CACHE_DIR": "/data3/junhaohu/.triton",
            "XDG_CACHE_HOME": "/data3/junhaohu/.cache",
            "VLLM_CACHE_ROOT": "/data3/junhaohu/.cache/vllm",
        }
    )
    try:
        for command in commands:
            subprocess.run(
                command,
                check=True,
                env=verifier_environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def append_event(root: Path, payload: dict[str, object]) -> None:
    with (root / "recovery_events.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_restart_source(root: Path) -> dict[str, str]:
    hashes = {}
    for relative, expected_digest in RESTART_SOURCE_HASHES.items():
        path = REPO / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise RuntimeError(f"unreviewed restart source: {relative} ({digest})")
        hashes[relative] = digest
    subprocess.run(
        [
            sys.executable,
            str(REPO / "training/audit_source_fidelity.py"),
            "--output",
            str(root / "supervisor_source_fidelity_audit.json"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return hashes


def validate_restart_data() -> dict[str, dict[str, object]]:
    from datasets import load_from_disk

    required_columns = {
        "input_ids",
        "chunk_ids",
        "cross_attention_mask",
        "labels",
        "token_count",
    }
    report = {}
    for name, (expected_rows, expected_fingerprint) in RESTART_DATASETS.items():
        path = DATASET_ROOT / f"{name}_meta-llama_Llama-3.1-8B-Instruct"
        dataset = load_from_disk(str(path))
        missing_columns = required_columns - set(dataset.column_names)
        if (
            len(dataset) != expected_rows
            or dataset._fingerprint != expected_fingerprint
            or missing_columns
        ):
            raise RuntimeError(
                f"unreviewed restart dataset {name}: rows={len(dataset)}, "
                f"fingerprint={dataset._fingerprint}, missing={sorted(missing_columns)}"
            )
        report[name] = {
            "rows": len(dataset),
            "fingerprint": dataset._fingerprint,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu-id", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--confirmations", type=int, default=3)
    parser.add_argument("--minimum-free-gib", type=int, default=300)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    stop_file = args.stop_file or root / "STOP_AUTORESTART"
    lock_path = root / "single_training_supervisor.lock"
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another single-training supervisor is active") from error

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

        tag, step = committed_checkpoint(root)
        if step >= FINAL_STEP and final_artifacts_complete(root):
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

        source_hashes = validate_restart_source(root)
        dataset_identities = validate_restart_data()
        recovery_id = time.strftime("%Y%m%dT%H%M%S%z")
        quarantined = quarantine_uncommitted_checkpoints(root, step, recovery_id)
        archive = rollback_loss_log(root, step, recovery_id)
        prior_stdout = rotate_stdout(root, recovery_id)
        event = {
            "time": time.time(),
            "event": "restart",
            "resume_tag": tag,
            "discarded_loss_archive": str(archive) if archive else None,
            "quarantined_uncommitted_checkpoints": [
                str(path) for path in quarantined
            ],
            "prior_stdout": str(prior_stdout) if prior_stdout else None,
            "free_gib": free_gib,
            "source_sha256": source_hashes,
            "datasets": dataset_identities,
        }
        append_event(root, event)
        environment = os.environ.copy()
        environment.update({"COMB_GPU_ID": str(args.gpu_id), "COMB_RESUME_TAG": tag})
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
