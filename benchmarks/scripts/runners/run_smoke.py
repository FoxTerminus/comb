"""Convenience smoke runner."""

from __future__ import annotations

import argparse

from benchmarks.scripts.runners.run_eval import run
from benchmarks.scripts.utils.config import load_run_config


MODEL_CONFIGS = {
    "mock": "benchmarks/configs/runs/smoke_mock.json",
    "combllama": "benchmarks/configs/runs/smoke_combllama.json",
    "llama": "benchmarks/configs/runs/smoke_llama.json",
    "yoco": "benchmarks/configs/runs/smoke_yoco.json",
    "sambay": "benchmarks/configs/runs/smoke_sambay.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke benchmark")
    parser.add_argument("--model", default="mock", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-ids-file", default=None)
    parser.add_argument("--retry-from", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(MODEL_CONFIGS[args.model])
    output_path = run(
        config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        resume=args.resume,
        limit=args.limit,
        only_ids_file=args.only_ids_file,
        retry_from=args.retry_from,
    )
    print(f"Wrote smoke run to {output_path}")


if __name__ == "__main__":
    main()
