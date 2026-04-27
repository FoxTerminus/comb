"""Run a local benchmark split through the configured adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.adapters import CombLlamaAdapter, MockAdapter
from benchmarks.scripts.schema import BenchmarkExample, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local benchmark examples")
    parser.add_argument("--config", default="benchmarks/configs/combllama_phase1.json")
    parser.add_argument("--split", default="phase1", help="Example split name, e.g. smoke or phase1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark names")
    parser.add_argument("--limit-per-benchmark", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Validate plumbing without loading the model")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--compression-ratio", type=float, default=None)
    parser.add_argument("--recent-window-tokens", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--cache-chunk-states", choices=["true", "false"], default=None)
    return parser.parse_args()


def load_examples(benchmarks: list[str], split: str, limit_per_benchmark: int) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for benchmark in benchmarks:
        path = REPO_ROOT / "benchmarks" / benchmark / f"{split}.jsonl"
        rows = read_jsonl(path)
        if limit_per_benchmark > 0:
            rows = rows[:limit_per_benchmark]
        for row in rows:
            examples.append(BenchmarkExample.from_dict(row))
    return examples


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = json.load(f)

    if args.compression_ratio is not None:
        config["model"]["compression_ratio"] = args.compression_ratio
    if args.recent_window_tokens is not None:
        config["model"]["recent_window_tokens"] = args.recent_window_tokens
    if args.chunk_size is not None:
        config["model"]["chunk_size"] = args.chunk_size
    if args.cache_chunk_states is not None:
        config["model"]["cache_chunk_states"] = args.cache_chunk_states == "true"

    benchmarks = list(config["benchmarks"])
    if args.benchmarks:
        benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    output_dir = Path(args.output_dir or config["paths"]["output_dir"])
    output_name = args.output_name or f"{args.split}_predictions.jsonl"
    max_new_tokens = args.max_new_tokens or int(config["generation"]["max_new_tokens"])
    examples = load_examples(benchmarks, args.split, args.limit_per_benchmark)

    adapter = MockAdapter() if args.mock else CombLlamaAdapter(config["model"])
    records = []
    for index, example in enumerate(examples, start=1):
        try:
            record = adapter.generate(example, max_new_tokens=max_new_tokens)
        except Exception as exc:
            record = MockAdapter(model_name="error").generate(example, max_new_tokens=0)
            record.error = f"{type(exc).__name__}: {exc}"
            record.exact_match = None
        row = record.to_dict()
        records.append(row)
        status = "ERR" if row.get("error") else "OK"
        print(
            f"[{index}/{len(examples)}] {status} "
            f"{example.benchmark}/{example.task}/{example.id}: {row['prediction']!r}"
        )

    output_path = output_dir / output_name
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
