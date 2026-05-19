#!/usr/bin/env python
"""Launch one LoCoMo shard per GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    script = Path(__file__).with_name("run.py")
    procs = []
    for shard_id, gpu in enumerate(gpus):
        cmd = [
            args.python,
            str(script),
            "--model",
            args.model,
            "--output-dir",
            args.output_dir,
            "--run-name",
            args.run_name,
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(len(gpus)),
            "--device",
            "cuda",
            "--dtype",
            args.dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repetition-penalty",
            str(args.repetition_penalty),
        ]
        if args.data_file:
            cmd.extend(["--data-file", args.data_file])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.overwrite:
            cmd.append("--overwrite")
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_dir = Path(args.output_dir) / args.run_name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (log_dir / f"{args.model}.shard{shard_id}-of-{len(gpus)}.log").open("w")
        print("Launching:", " ".join(cmd), f"on GPU {gpu}")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(code != 0 for code in codes):
        raise SystemExit(f"At least one shard failed: {codes}")


if __name__ == "__main__":
    main()
