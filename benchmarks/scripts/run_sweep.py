"""Run a CombLlama compression-ratio sweep over a local benchmark split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.adapters import CombLlamaAdapter, MockAdapter
from benchmarks.scripts.run_examples import load_examples
from benchmarks.scripts.schema import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CombLlama compression sweep")
    parser.add_argument("--config", default="benchmarks/configs/combllama_phase3_sweep.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark names")
    parser.add_argument("--compression-ratios", default=None, help="Comma-separated ratios")
    parser.add_argument("--limit-per-benchmark", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--cache-chunk-states", choices=["true", "false"], default=None)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def ratio_label(ratio: float) -> str:
    return str(ratio).replace(".", "p")


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = json.load(f)

    if args.cache_chunk_states is not None:
        config["model"]["cache_chunk_states"] = args.cache_chunk_states == "true"

    benchmarks = list(config["benchmarks"])
    if args.benchmarks:
        benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    if args.compression_ratios:
        ratios = [float(item) for item in args.compression_ratios.split(",") if item.strip()]
    else:
        ratios = [float(item) for item in config["sweep"]["compression_ratios"]]

    retention_policy = str(config["model"].get("retention_policy", "all_encoder_chunks"))
    if retention_policy != "all_encoder_chunks":
        raise ValueError("Only all_encoder_chunks is supported")

    limit = args.limit_per_benchmark
    if limit is None:
        limit = int(config.get("sweep", {}).get("default_limit_per_benchmark", 0))

    max_new_tokens = args.max_new_tokens or int(config["generation"]["max_new_tokens"])
    output_dir = Path(args.output_dir or config["paths"]["output_dir"])
    examples = load_examples(benchmarks, args.split, limit)

    adapter = MockAdapter() if args.mock else CombLlamaAdapter(config["model"])
    manifest = []

    for ratio in ratios:
        if hasattr(adapter, "compression_ratio"):
            adapter.compression_ratio = ratio
        records = []
        print(f"Running retention_policy={retention_policy} compression_ratio={ratio} on {len(examples)} examples")
        for index, example in enumerate(examples, start=1):
            try:
                record = adapter.generate(example, max_new_tokens=max_new_tokens)
            except Exception as exc:
                record = MockAdapter(model_name="error").generate(example, max_new_tokens=0)
                record.error = f"{type(exc).__name__}: {exc}"
                record.exact_match = None
                record.compression_ratio = ratio
            row = record.to_dict()
            row["compression_ratio"] = ratio
            row["retention_policy"] = retention_policy
            records.append(row)
            status = "ERR" if row.get("error") else "OK"
            print(
                f"[policy={retention_policy} ratio={ratio} {index}/{len(examples)}] {status} "
                f"{example.benchmark}/{example.task}/{example.id}: {row['prediction']!r}"
            )

        output_path = output_dir / (
            f"{args.split}_predictions_{retention_policy}_cr{ratio_label(ratio)}.jsonl"
        )
        write_jsonl(output_path, records)
        manifest.append(
            {
                "retention_policy": retention_policy,
                "compression_ratio": ratio,
                "path": str(output_path),
                "num_examples": len(records),
            }
        )
        print(f"Wrote {len(records)} records to {output_path}")

    manifest_path = output_dir / f"{args.split}_sweep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote sweep manifest to {manifest_path}")


if __name__ == "__main__":
    main()
