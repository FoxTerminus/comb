#!/usr/bin/env python3
"""Restore the small DeepSpeed model-state envelope for a universal checkpoint.

DeepSpeed's universal conversion stores trainable weights and optimizer states
under ``zero/`` but still requires an ``mp_rank_*_model_states.pt`` envelope to
enter the loader and recover global/client state.  This tool creates only that
envelope; it never rewrites universal parameter files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


EXPECTED_STEP = 20_000
EXPECTED_LR = 4.988124140517565e-05
EXPECTED_PARAMETER_DIRECTORIES = 176
OFFICIAL_COMMIT = "25bb50823ab5998d0caa55014ec07771ca9fba9a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    marker_path = checkpoint / "comb_universal_conversion_complete.json"
    optimizer_path = checkpoint / "zero/optimizer_state.pt"
    target = checkpoint / "mp_rank_00_model_states.pt"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing metadata: {target}")
    marker = json.loads(marker_path.read_text())
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    parameter_directories = sorted(
        path for path in (checkpoint / "zero").iterdir() if path.is_dir()
    )
    lr = float(optimizer["param_groups"][0]["lr"])
    failures = []
    if int(marker.get("file_count", -1)) != 706:
        failures.append("conversion_file_count")
    if len(parameter_directories) != EXPECTED_PARAMETER_DIRECTORIES:
        failures.append("parameter_directory_count")
    if abs(lr - EXPECTED_LR) > 1e-15:
        failures.append("optimizer_learning_rate")
    if str(optimizer.get("ds_version")) != "0.17.2":
        failures.append("deepspeed_version")
    if failures:
        raise RuntimeError(f"universal checkpoint validation failed: {failures}")

    envelope = {
        # Universal loading deliberately skips this module dictionary and
        # restores all trainable low-precision parameters from zero/*/fp32.pt.
        "module": {},
        "buffer_names": [],
        "optimizer": None,
        "param_shapes": [],
        "frozen_param_shapes": None,
        "shared_params": {},
        "frozen_param_fragments": None,
        "lr_scheduler": {"last_batch_iteration": EXPECTED_STEP - 1},
        "data_sampler": None,
        "random_ltd": None,
        "sparse_tensor_module_names": set(),
        "skipped_steps": 0,
        "global_steps": EXPECTED_STEP,
        "global_samples": EXPECTED_STEP * 32,
        "dp_world_size": 1,
        "mp_world_size": 1,
        "ds_version": "0.17.2",
        "optimizer_step": EXPECTED_STEP,
        "dataset_index": 1,
        "bucket_index": 0,
        "next_batch_index": 63_706,
        "git_commit": OFFICIAL_COMMIT,
        "universal_checkpoint_info": marker["universal_checkpoint_info"],
    }
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    torch.save(envelope, temporary)
    os.replace(temporary, target)
    report = {
        "checkpoint": str(checkpoint),
        "created_metadata": str(target),
        "metadata_sha256": sha256(target),
        "universal_optimizer_state_sha256": sha256(optimizer_path),
        "parameter_directories": len(parameter_directories),
        "optimizer_learning_rate": lr,
        "global_steps": EXPECTED_STEP,
        "resume_position": {
            "dataset_index": 1,
            "bucket_index": 0,
            "next_batch_index": 63_706,
        },
        "passed": True,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
