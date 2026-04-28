"""Summarize and sanity-check SambaY training diagnostics CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize SambaY training_diagnostics.csv")
    parser.add_argument(
        "path",
        nargs="?",
        default="/data3/junhaohu/comb/baselines/SambaY/training/training_diagnostics.csv",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1000.0)
    parser.add_argument("--require-supervised-tokens", action="store_true", default=True)
    parser.add_argument("--allow-zero-supervised-tokens", action="store_false", dest="require_supervised_tokens")
    return parser.parse_args()


def parse_float(row, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def parse_int(row, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No diagnostics rows found in {path}")

    losses = [parse_float(row, "loss") for row in rows]
    grad_norms = [parse_float(row, "grad_norm") for row in rows]
    supervised_tokens = [parse_int(row, "supervised_tokens") for row in rows]
    total_tokens = [parse_int(row, "total_tokens") for row in rows]
    max_seqlen = [parse_int(row, "max_seqlen_q") for row in rows]
    peak_memory = [parse_float(row, "peak_memory_gb") for row in rows if row.get("peak_memory_gb", "") != ""]

    bad_losses = [value for value in losses if not math.isfinite(value)]
    bad_grads = [value for value in grad_norms if not math.isfinite(value) or value > args.max_grad_norm]
    zero_supervised = sum(1 for value in supervised_tokens if value <= 0)

    if bad_losses:
        raise RuntimeError(f"Found non-finite losses: {bad_losses[:5]}")
    if bad_grads:
        raise RuntimeError(f"Found invalid or too-large grad norms: {bad_grads[:5]}")
    if args.require_supervised_tokens and zero_supervised:
        raise RuntimeError(f"Found {zero_supervised} rows with zero supervised tokens.")

    first = rows[0]
    last = rows[-1]
    print(f"path={path}")
    print(f"rows={len(rows)}")
    print(f"first_step={first.get('global_step')} last_step={last.get('global_step')}")
    print(f"first_loss={losses[0]} last_loss={losses[-1]} min_loss={min(losses)} max_loss={max(losses)}")
    print(f"max_grad_norm={max(grad_norms)}")
    print(f"total_tokens={sum(total_tokens)} supervised_tokens={sum(supervised_tokens)}")
    print(f"max_seqlen_q={max(max_seqlen)}")
    if peak_memory:
        print(f"peak_memory_gb={max(peak_memory)}")
    print("diagnostics_ok=True")


if __name__ == "__main__":
    main()
