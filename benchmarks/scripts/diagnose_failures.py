"""Diagnose prompt packing and repetitive generation failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.prompting import inspect_combllama_packing, render_qa_prompt
from benchmarks.scripts.run_examples import load_examples
from benchmarks.scripts.schema import BenchmarkExample, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose benchmark failure modes")
    parser.add_argument("--config", default="benchmarks/configs/combllama_phase3_sweep.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark names")
    parser.add_argument("--compression-ratios", default=None, help="Comma-separated ratios")
    parser.add_argument("--predictions-manifest", required=True)
    parser.add_argument("--output-dir", default="benchmarks/reports/phase7_failure_analysis")
    return parser.parse_args()


def encode_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> list[int]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return tokenizer(prompt, add_special_tokens=False).input_ids


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    last_start = len(haystack) - len(needle)
    for idx in range(last_start + 1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return None


def locate_answer_region(
    tokenizer: Any,
    prompt: str,
    token_ids: list[int],
    answer: str | None,
    *,
    history_tokens: int,
    kept_chunk_tokens: int,
    decoder_tokens: int,
    chunk_size: int | None = None,
    kept_chunk_indices: list[int] | None = None,
) -> dict[str, Any]:
    if not answer:
        return {
            "answer_token_start": "",
            "answer_token_length": "",
            "answer_token_region": "no_answer",
        }
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    start = find_subsequence(token_ids, answer_ids)
    if start is None:
        char_start = prompt.find(answer)
        if char_start >= 0:
            plain_prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            template_overhead = max(0, len(token_ids) - plain_prompt_tokens)
            start = len(tokenizer(prompt[:char_start], add_special_tokens=False).input_ids) + template_overhead
    if start is None:
        return {
            "answer_token_start": "",
            "answer_token_length": len(answer_ids),
            "answer_token_region": "not_found",
        }

    decoder_start = max(0, len(token_ids) - decoder_tokens)
    if start >= decoder_start:
        region = "decoder"
    elif start < history_tokens:
        if chunk_size is not None and kept_chunk_indices is not None:
            answer_chunk = start // max(1, chunk_size)
            region = "kept_chunk" if answer_chunk in set(kept_chunk_indices) else "dropped_chunk"
        else:
            kept_start = max(0, history_tokens - kept_chunk_tokens)
            region = "kept_chunk" if start >= kept_start else "dropped_chunk"
    else:
        region = "boundary"

    return {
        "answer_token_start": start,
        "answer_token_length": len(answer_ids),
        "answer_token_region": region,
    }


def load_manifest_predictions(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in manifest:
        for row in read_jsonl(item["path"]):
            rows.append(row)
    return rows


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def repeated_substring_unit(text: str, max_unit: int = 12) -> str | None:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 12:
        return None
    for size in range(1, min(max_unit, len(compact) // 2) + 1):
        unit = compact[:size]
        if unit and unit * (len(compact) // size) == compact[: size * (len(compact) // size)]:
            coverage = size * (len(compact) // size) / len(compact)
            if coverage >= 0.85 and len(compact) // size >= 4:
                return unit
    return None


def repetition_metrics(prediction: str) -> dict[str, Any]:
    tokens = normalized_tokens(prediction)
    if not tokens:
        return {
            "prediction_chars": len(prediction),
            "prediction_tokens": 0,
            "unique_token_ratio": 0.0,
            "max_token_fraction": 0.0,
            "max_token_run": 0,
            "repeated_substring_unit": None,
            "collapse_flag": False,
        }

    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1

    max_run = 1
    current_run = 1
    for prev, token in zip(tokens, tokens[1:]):
        if token == prev:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    unit = repeated_substring_unit(prediction)
    unique_token_ratio = len(counts) / len(tokens)
    max_token_fraction = max(counts.values()) / len(tokens)
    collapse_flag = bool(
        unit
        or (len(tokens) >= 8 and unique_token_ratio <= 0.25 and max_token_fraction >= 0.50)
        or max_run >= 8
    )

    return {
        "prediction_chars": len(prediction),
        "prediction_tokens": len(tokens),
        "unique_token_ratio": unique_token_ratio,
        "max_token_fraction": max_token_fraction,
        "max_token_run": max_run,
        "repeated_substring_unit": unit,
        "collapse_flag": collapse_flag,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = list(config["benchmarks"])
    if args.benchmarks:
        benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    if args.compression_ratios:
        ratios = [float(item) for item in args.compression_ratios.split(",") if item.strip()]
    else:
        ratios = [float(item) for item in config["sweep"]["compression_ratios"]]
    retention_policy = str(model_config.get("retention_policy", "all_encoder_chunks"))
    if retention_policy != "all_encoder_chunks":
        raise ValueError("Only all_encoder_chunks is supported")

    from transformers import AutoTokenizer

    tokenizer_path = str(model_config.get("tokenizer_path", model_config["model_path"]))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    use_chat_template = bool(model_config.get("use_chat_template", True))
    chunk_size = int(model_config.get("chunk_size", 1024))
    recent_window_tokens = int(model_config.get("recent_window_tokens", 1024))

    examples = load_examples(benchmarks, args.split, limit_per_benchmark=0)
    example_by_id: dict[str, BenchmarkExample] = {example.id: example for example in examples}

    prompt_rows = []
    packing_rows = []
    for example in examples:
        prompt = render_qa_prompt(example)
        token_ids = encode_prompt(tokenizer, prompt, use_chat_template)
        prompt_rows.append(
            {
                "id": example.id,
                "benchmark": example.benchmark,
                "task": example.task,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_chars": len(prompt),
                "encoded_prompt_tokens": len(token_ids),
                "rendered_prompt": prompt,
            }
        )
        for ratio in ratios:
            diagnostics = inspect_combllama_packing(
                token_ids,
                chunk_size=chunk_size,
                recent_window_tokens=recent_window_tokens,
                compression_ratio=ratio,
                retention_policy=retention_policy,
            )
            answer_region = locate_answer_region(
                tokenizer,
                prompt,
                token_ids,
                example.answer,
                history_tokens=diagnostics.history_tokens,
                kept_chunk_tokens=diagnostics.kept_chunk_tokens,
                decoder_tokens=diagnostics.decoder_tokens,
                chunk_size=diagnostics.chunk_size,
                kept_chunk_indices=diagnostics.kept_chunk_indices,
            )
            packing_rows.append(
                {
                    "id": example.id,
                    "benchmark": example.benchmark,
                    "task": example.task,
                    **asdict(diagnostics),
                    **answer_region,
                }
            )

    write_jsonl(output_dir / "rendered_prompts.jsonl", prompt_rows)
    write_jsonl(output_dir / "packing_diagnostics.jsonl", packing_rows)

    predictions = load_manifest_predictions(Path(args.predictions_manifest))
    packing_by_key = {
        (row["id"], str(row["retention_policy"]), float(row["compression_ratio"])): row
        for row in packing_rows
    }

    prediction_rows = []
    for row in predictions:
        ratio = float(row.get("compression_ratio", 1.0))
        retention_policy = str(row.get("retention_policy", model_config.get("retention_policy", "all_encoder_chunks")))
        packing = packing_by_key.get((row["id"], retention_policy, ratio), {})
        metrics = repetition_metrics(str(row.get("prediction", "")))
        prediction_rows.append(
            {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "task": row["task"],
                "compression_ratio": ratio,
                "retention_policy": retention_policy,
                "answer": row.get("answer"),
                "prediction": row.get("prediction", ""),
                "generated_tokens": row.get("generated_tokens"),
                "prompt_tokens": row.get("prompt_tokens"),
                "chunk_tokens": row.get("chunk_tokens"),
                "decoder_tokens": row.get("decoder_tokens"),
                "kept_chunks": packing.get("kept_chunks"),
                "dropped_chunks": packing.get("dropped_chunks"),
                "kept_chunk_tokens": packing.get("kept_chunk_tokens"),
                "dropped_chunk_tokens": packing.get("dropped_chunk_tokens"),
                "answer_token_region": packing.get("answer_token_region"),
                "answer_token_start": packing.get("answer_token_start"),
                **metrics,
            }
        )

    write_jsonl(output_dir / "prediction_diagnostics.jsonl", prediction_rows)

    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        groups[(str(row["retention_policy"]), float(row["compression_ratio"]), row["benchmark"])].append(row)

    summary_rows = []
    for (retention_policy, ratio, benchmark), items in sorted(groups.items()):
        summary_rows.append(
            {
                "retention_policy": retention_policy,
                "compression_ratio": ratio,
                "benchmark": benchmark,
                "num_examples": len(items),
                "collapse_rate": mean([float(item["collapse_flag"]) for item in items]),
                "avg_unique_token_ratio": mean([float(item["unique_token_ratio"]) for item in items]),
                "avg_max_token_fraction": mean([float(item["max_token_fraction"]) for item in items]),
                "avg_max_token_run": mean([float(item["max_token_run"]) for item in items]),
                "avg_kept_chunk_tokens": mean(
                    [float(item["kept_chunk_tokens"]) for item in items if item["kept_chunk_tokens"] != ""]
                ),
                "avg_dropped_chunk_tokens": mean(
                    [float(item["dropped_chunk_tokens"]) for item in items if item["dropped_chunk_tokens"] != ""]
                ),
                "answer_in_dropped_rate": mean(
                    [float(item["answer_token_region"] == "dropped_chunk") for item in items]
                ),
                "answer_in_kept_or_decoder_rate": mean(
                    [
                        float(item["answer_token_region"] in {"kept_chunk", "decoder"})
                        for item in items
                    ]
                ),
            }
        )

    write_csv(
        output_dir / "collapse_summary.csv",
        summary_rows,
        [
            "compression_ratio",
            "retention_policy",
            "benchmark",
            "num_examples",
            "collapse_rate",
            "avg_unique_token_ratio",
            "avg_max_token_fraction",
            "avg_max_token_run",
            "avg_kept_chunk_tokens",
            "avg_dropped_chunk_tokens",
            "answer_in_dropped_rate",
            "answer_in_kept_or_decoder_rate",
        ],
    )

    markdown = [
        "# Failure Diagnostics",
        "",
        "## Outputs",
        "",
        "```text",
        "rendered_prompts.jsonl",
        "packing_diagnostics.jsonl",
        "prediction_diagnostics.jsonl",
        "collapse_summary.csv",
        "```",
        "",
        "## Collapse Summary",
        "",
        "| Policy | Compression | Benchmark | Examples | Collapse Rate | Answer Dropped | Answer Kept/Decoder |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        markdown.append(
            "| {retention_policy} | {compression_ratio} | {benchmark} | {num_examples} | {collapse_rate:.3f} | {answer_in_dropped_rate:.3f} | {answer_in_kept_or_decoder_rate:.3f} |".format(
                **row
            )
        )
    markdown.extend(
        [
            "",
            "## Notes",
            "",
            "A collapse flag is raised for repeated compact substrings, very low token diversity with a dominant token, or long same-token runs.",
            "",
            "Packing diagnostics reflect the current benchmark policy: compressed ratios retain the most recent history chunks and drop earlier history chunks.",
        ]
    )
    (output_dir / "failure_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote failure diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
