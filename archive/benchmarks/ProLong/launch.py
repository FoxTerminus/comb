#!/usr/bin/env python
"""Launch sharded ProLong validation evaluation, one process per GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from eval import MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="/data3/junhaohu/comb/benchmarks/ProLong/results")
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default="validation*.bin")
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-stride", type=int, default=20)
    parser.add_argument("--comb-ctx-len", type=int, default=65536)
    parser.add_argument("--comb-target-len", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")

    script = Path(__file__).with_name("eval.py")
    log_dir = Path(args.output_dir) / args.run_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
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
            "--device",
            "cuda",
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--pattern",
            args.pattern,
            "--file-start",
            str(args.file_start),
            "--file-stride",
            str(args.file_stride),
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(len(gpus)),
            "--comb-ctx-len",
            str(args.comb_ctx_len),
            "--comb-target-len",
            str(args.comb_target_len),
        ]
        if args.data_dir:
            cmd.extend(["--data-dir", args.data_dir])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = log_dir / f"{args.model}.shard{shard_id}-of-{len(gpus)}.log"
        log_file = log_path.open("w")
        print("Launching:", " ".join(cmd), f"on GPU {gpu}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))

    codes = [p.wait() for p in procs]
    if any(code != 0 for code in codes):
        raise SystemExit(f"At least one shard failed: {codes}")


if __name__ == "__main__":
    main()
