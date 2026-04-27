"""Run Phase 0 smoke examples through the benchmark adapter."""

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
    parser = argparse.ArgumentParser(description="Run CombLlama benchmark smoke examples")
    parser.add_argument("--config", default="benchmarks/configs/combllama_smoke.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mock", action="store_true", help="Validate plumbing without loading the model")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def load_examples(benchmarks: list[str]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for benchmark in benchmarks:
        path = REPO_ROOT / "benchmarks" / benchmark / "smoke.jsonl"
        for row in read_jsonl(path):
            examples.append(BenchmarkExample.from_dict(row))
    return examples


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(args.output_dir or config["paths"]["output_dir"])
    max_new_tokens = args.max_new_tokens or int(config["generation"]["max_new_tokens"])
    examples = load_examples(list(config["benchmarks"]))

    adapter = MockAdapter() if args.mock else CombLlamaAdapter(config["model"])
    records = []
    for example in examples:
        try:
            record = adapter.generate(example, max_new_tokens=max_new_tokens)
        except Exception as exc:
            record = MockAdapter(model_name="error").generate(example, max_new_tokens=0)
            record.error = f"{type(exc).__name__}: {exc}"
            record.exact_match = None
        records.append(record.to_dict())
        print(f"{example.benchmark}/{example.task}/{example.id}: {records[-1]['prediction']!r}")

    write_jsonl(output_dir / "smoke_predictions.jsonl", records)
    print(f"Wrote {len(records)} records to {output_dir / 'smoke_predictions.jsonl'}")


if __name__ == "__main__":
    main()
