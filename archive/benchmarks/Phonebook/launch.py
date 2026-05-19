#!/usr/bin/env python
"""Launch Phonebook benchmark shards, one process per GPU."""

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
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="phonebook_32k")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-pairs", type=int, default=1850)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-mode", choices=["uniform", "early", "middle", "late"], default="uniform")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")

    script = Path(__file__).with_name("run.py")
    log_dir = Path(args.output_dir) / args.run_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard_id, gpu in enumerate(gpus):
        cmd = [
            args.python,
            str(script),
            "--model", args.model,
            "--output-dir", args.output_dir,
            "--run-name", args.run_name,
            "--num-samples", str(args.num_samples),
            "--num-pairs", str(args.num_pairs),
            "--seed", str(args.seed),
            "--position-mode", args.position_mode,
            "--device", "cuda",
            "--dtype", args.dtype,
            "--max-new-tokens", str(args.max_new_tokens),
            "--repetition-penalty", str(args.repetition_penalty),
            "--shard-id", str(shard_id),
            "--num-shards", str(len(gpus)),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = log_dir / f"{args.model}.shard{shard_id}-of-{len(gpus)}.log"
        log_file = log_path.open("w")
        print("Launching:", " ".join(cmd), f"on GPU {gpu}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))

    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise SystemExit(f"At least one shard failed: {codes}")


if __name__ == "__main__":
    main()

