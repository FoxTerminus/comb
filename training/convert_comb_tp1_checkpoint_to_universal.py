#!/usr/bin/env python3
"""Convert a native Comb DeepSpeed checkpoint to universal format.

Both the released TP=1 checkpoints and native true-TP checkpoints may omit
``universal_checkpoint_info``.  This wrapper injects the verified Comb
parameter-layout metadata in memory, so the source checkpoint is never
rewritten or modified in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

from deepspeed.checkpoint.constants import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint import ds_to_universal

from training.true_tp_comb_adapter import comb_universal_checkpoint_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--num-extract-workers", type=int, default=2)
    parser.add_argument("--num-merge-workers", type=int, default=1)
    parser.add_argument("--keep-temp-folder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input_folder.resolve()
    destination = args.output_folder.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing universal checkpoint: {destination}"
        )
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination checkpoint directories must be disjoint")

    def inject_comb_state(ds_checkpoint) -> None:
        expected = comb_universal_checkpoint_info()
        existing = ds_checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO)
        if existing is not None and existing != expected:
            raise RuntimeError("source checkpoint contains conflicting universal metadata")
        ds_checkpoint.global_state[UNIVERSAL_CHECKPOINT_INFO] = expected

    ds_to_universal._inject_missing_state = inject_comb_state
    ds_to_universal.main(
        SimpleNamespace(
            input_folder=str(source),
            output_folder=str(destination),
            num_extract_workers=args.num_extract_workers,
            num_merge_workers=args.num_merge_workers,
            keep_temp_folder=args.keep_temp_folder,
            strict=True,
            inject_missing_state=True,
        )
    )
    files = [path for path in destination.rglob("*") if path.is_file()]
    manifest = {
        "completed_unix": time.time(),
        "source": str(source),
        "destination": str(destination),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "universal_checkpoint_info": comb_universal_checkpoint_info(),
    }
    (destination / "comb_universal_conversion_complete.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
