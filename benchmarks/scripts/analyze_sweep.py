"""Generate Phase 4 analysis tables and a Markdown report from a sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.metrics import contains_match, exact_match, token_f1
from benchmarks.scripts.schema import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CombLlama benchmark sweep results")
    parser.add_argument("--manifest", default="benchmarks/results/phase3_combllama_dev_sweep_mnt8/dev_sweep_manifest.json")
    parser.add_argument("--output-dir", default="benchmarks/reports/phase4")
    parser.add_argument("--title", default="CombLlama Phase 4 Dev Sweep Analysis")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in manifest:
        rows.extend(read_jsonl(item["path"]))
    return rows


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    answer_items = [item for item in items if item.get("answer") is not None]
    memory_values = [float(item["peak_memory_gb"]) for item in items if item.get("peak_memory_gb") is not None]
    return {
        "num_examples": len(items),
        "num_errors": sum(1 for item in items if item.get("error")),
        "exact_match": mean([float(exact_match(item.get("prediction"), item.get("answer"))) for item in answer_items]),
        "contains_match": mean([float(contains_match(item.get("prediction"), item.get("answer"))) for item in answer_items]),
        "token_f1": mean([token_f1(item.get("prediction"), item.get("answer")) or 0.0 for item in answer_items]),
        "avg_prompt_tokens": mean([float(item["prompt_tokens"]) for item in items]),
        "avg_chunk_tokens": mean([float(item["chunk_tokens"]) for item in items]),
        "avg_decoder_tokens": mean([float(item["decoder_tokens"]) for item in items]),
        "avg_generated_tokens": mean([float(item["generated_tokens"]) for item in items]),
        "avg_peak_memory_gb": mean(memory_values),
        "max_peak_memory_gb": max(memory_values) if memory_values else 0.0,
        "avg_prefill_latency_s": mean([float(item["prefill_latency_s"]) for item in items]),
        "avg_decode_latency_s": mean([float(item["decode_latency_s"]) for item in items]),
        "avg_tokens_per_second": mean(
            [float(item["tokens_per_second"]) for item in items if item.get("tokens_per_second") is not None]
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_rows_by_ratio(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[float(row.get("compression_ratio", 1.0))].append(row)

    output = []
    baseline = None
    for ratio, items in sorted(groups.items(), reverse=True):
        summary = summarize_group(items)
        if ratio == 1.0:
            baseline = summary
        row = {"compression_ratio": ratio, **summary}
        output.append(row)

    if baseline:
        for row in output:
            row["max_memory_saving_vs_1p0"] = (
                (baseline["max_peak_memory_gb"] - row["max_peak_memory_gb"]) / baseline["max_peak_memory_gb"]
            )
            row["avg_prefill_speedup_vs_1p0"] = baseline["avg_prefill_latency_s"] / max(row["avg_prefill_latency_s"], 1e-12)
            row["avg_decode_speedup_vs_1p0"] = baseline["avg_decode_latency_s"] / max(row["avg_decode_latency_s"], 1e-12)
    return output


def make_rows_by_benchmark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row.get("compression_ratio", 1.0)), str(row["benchmark"]))].append(row)
    output = []
    for (ratio, benchmark), items in sorted(groups.items()):
        output.append({"compression_ratio": ratio, "benchmark": benchmark, **summarize_group(items)})
    return output


def make_cache_report(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# CombLlama KV Cache Feasibility Note",
                "",
                "Current benchmark inference caches CombLlama chunk encoder outputs once per prompt and reuses those cross-attention states during decoding.",
                "",
                "What is already implemented:",
                "",
                "1. Historical prompt tokens are packed into chunks once.",
                "2. `model.chunk_model(...)` is run once during prefill.",
                "3. The resulting cross-attention K/V states are reused for every generated token.",
                "4. This avoids repeatedly encoding 10K-100K historical chunk tokens.",
                "",
                "What is still missing:",
                "",
                "1. `CombLlamaTextModel.forward` creates a `DynamicCache` when `use_cache=True`, but the custom self-attention path `_self_attn_forward` computes Q/K/V from the current input and calls `flash_attn_varlen_func` directly.",
                "2. `_self_attn_forward` does not update `past_key_values` for self-attention layers.",
                "3. `_self_attn_forward` also does not read previous self-attention keys/values during decode.",
                "4. Cross-attention cache support exists for compressed chunk states, but that alone is insufficient for autoregressive token-by-token decoding because recent decoder self-attention still needs cached self K/V.",
                "",
                "Conclusion:",
                "",
                "The benchmark can now run `max_new_tokens=32` dev sweeps using cached chunk states. Do not switch to `use_cache=True` for self-attention until self-attention cache update and decode semantics are implemented and verified against no-cache generation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_markdown_report(
    path: Path,
    title: str,
    by_ratio: list[dict[str, Any]],
    by_benchmark: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    max_generated = max(int(row.get("generated_tokens", 0)) for row in rows) if rows else 0
    avg_generated = mean([float(row.get("generated_tokens", 0)) for row in rows])
    ratio_lines = [
        "| Compression | Examples | Contains | F1 | Avg Chunk Tokens | Max Mem GB | Prefill s | Decode s | Mem Saving |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_ratio:
        ratio_lines.append(
            "| {compression_ratio} | {num_examples} | {contains_match:.3f} | {token_f1:.3f} | "
            "{avg_chunk_tokens:.1f} | {max_peak_memory_gb:.2f} | {avg_prefill_latency_s:.3f} | "
            "{avg_decode_latency_s:.3f} | {max_memory_saving_vs_1p0:.1%} |".format(**row)
        )

    benchmark_lines = [
        "| Compression | Benchmark | Examples | Contains | F1 | Max Mem GB | Decode s | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_benchmark:
        benchmark_lines.append(
            "| {compression_ratio} | {benchmark} | {num_examples} | {contains_match:.3f} | "
            "{token_f1:.3f} | {max_peak_memory_gb:.2f} | {avg_decode_latency_s:.3f} | {num_errors} |".format(**row)
        )

    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "## Scope",
                "",
                "This report analyzes the CombLlama-only Phase 3 dev sweep. YOCO is excluded until its training finishes.",
                "",
                "The sweep covers RULER, LongBench, SCBench, LongCodeBench, and LoCoMo with compression ratios `1.0`, `0.5`, `0.25`, and `0.125`.",
                "",
                f"This run generated up to `{max_generated}` tokens per example, with average generated tokens `{avg_generated:.1f}`.",
                "",
                "Quality metrics are still diagnostic because official benchmark-specific scoring has not yet been plugged in.",
                "",
                "## Overall By Compression Ratio",
                "",
                *ratio_lines,
                "",
                "## By Benchmark",
                "",
                *benchmark_lines,
                "",
                "## Interpretation",
                "",
                "The efficiency trend is clear: retaining fewer historical chunks reduces memory and latency monotonically.",
                "",
                "Quality degrades sharply when the retained compressed context is reduced. This is visible in both the generated text and the near-zero contains/F1 metrics at lower ratios.",
                "",
                "The current low exact-match scores should not be treated as final model quality because official task metrics have not yet been plugged in and prompt/task formatting is still preliminary.",
                "",
                "## Next Engineering Step",
                "",
                "Implement and verify self-attention KV cache reuse in CombLlama inference. Chunk encoder caching now makes `max_new_tokens=32` dev sweeps feasible, but longer generation and larger full benchmark runs still re-run the recent decoder window for each generated token.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.manifest))
    by_ratio = make_rows_by_ratio(rows)
    by_benchmark = make_rows_by_benchmark(rows)

    ratio_fields = [
        "compression_ratio",
        "num_examples",
        "num_errors",
        "exact_match",
        "contains_match",
        "token_f1",
        "avg_prompt_tokens",
        "avg_chunk_tokens",
        "avg_decoder_tokens",
        "avg_generated_tokens",
        "avg_peak_memory_gb",
        "max_peak_memory_gb",
        "avg_prefill_latency_s",
        "avg_decode_latency_s",
        "avg_tokens_per_second",
        "max_memory_saving_vs_1p0",
        "avg_prefill_speedup_vs_1p0",
        "avg_decode_speedup_vs_1p0",
    ]
    benchmark_fields = [
        "compression_ratio",
        "benchmark",
        "num_examples",
        "num_errors",
        "exact_match",
        "contains_match",
        "token_f1",
        "avg_prompt_tokens",
        "avg_chunk_tokens",
        "avg_decoder_tokens",
        "avg_generated_tokens",
        "avg_peak_memory_gb",
        "max_peak_memory_gb",
        "avg_prefill_latency_s",
        "avg_decode_latency_s",
        "avg_tokens_per_second",
    ]

    write_csv(output_dir / "overall_by_ratio.csv", by_ratio, ratio_fields)
    write_csv(output_dir / "by_benchmark.csv", by_benchmark, benchmark_fields)
    make_cache_report(output_dir / "kv_cache_feasibility.md")
    make_markdown_report(output_dir / "final_report.md", args.title, by_ratio, by_benchmark, rows)
    print(f"Wrote Phase 4 analysis to {output_dir}")


if __name__ == "__main__":
    main()
