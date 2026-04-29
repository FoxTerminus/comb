"""Tokenizer-only length audit for converted benchmark data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.scripts.adapters.base import render_prompt
from benchmarks.scripts.schema import BENCHMARKS, BenchmarkExample
from benchmarks.scripts.utils.config import repo_path
from benchmarks.scripts.utils.data import load_examples
from benchmarks.scripts.utils.io import read_json, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_THRESHOLDS = [4096, 8192, 16384, 32768, 65536, 131072]


class TokenCounter:
    def __init__(self, tokenizer_path: str | None = None, *, use_chat_template: bool = True, require_tokenizer: bool = False) -> None:
        self.tokenizer_path = tokenizer_path
        self.use_chat_template = use_chat_template
        self.tokenizer = None
        self.name = "whitespace"
        self.error: str | None = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.name = tokenizer_path
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                if require_tokenizer:
                    raise

    def count_text(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def count_prompt(self, prompt: str) -> int:
        if self.tokenizer is None:
            return len(prompt.split())
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return len(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        return len(self.tokenizer(prompt, add_special_tokens=False).input_ids)


def length_bucket(prompt_tokens: int, thresholds: list[int]) -> str:
    for threshold in thresholds:
        if prompt_tokens <= threshold:
            return f"<= {threshold}"
    return f"> {thresholds[-1]}"


def load_model_limits(model_config_paths: list[str]) -> dict[str, dict[str, Any]]:
    limits: dict[str, dict[str, Any]] = {}
    for config_path in model_config_paths:
        config = read_json(repo_path(config_path))
        name = str(config.get("name", Path(config_path).stem))
        adapter = str(config.get("adapter", name))
        if adapter == "mock":
            continue
        limits[name] = {
            "max_context_tokens": int(config.get("max_context_tokens", 131072)),
            "kv_cache_policy": config.get("kv_cache_policy", "unknown"),
            "chunk_size": config.get("chunk_size"),
            "recent_window_tokens": config.get("recent_window_tokens"),
        }
    return limits


def audit_examples(
    examples: list[BenchmarkExample],
    *,
    token_counter: TokenCounter,
    thresholds: list[int],
    model_limits: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = render_prompt(example)
        context_tokens = token_counter.count_text(example.context)
        question_tokens = token_counter.count_text(example.question)
        answer_tokens = 0 if example.answer is None else token_counter.count_text(example.answer)
        prompt_tokens = token_counter.count_prompt(prompt)
        over_limit_models = [
            model_name
            for model_name, limit in model_limits.items()
            if prompt_tokens > int(limit["max_context_tokens"])
        ]
        row = {
            "id": example.id,
            "benchmark": example.benchmark,
            "task": example.task,
            "split": example.split,
            "context_tokens": context_tokens,
            "question_tokens": question_tokens,
            "answer_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "length_bucket": length_bucket(prompt_tokens, thresholds),
            "over_limit_models": ",".join(over_limit_models),
        }
        for threshold in thresholds:
            row[f"over_{threshold}"] = prompt_tokens > threshold
        if "combllama" in model_limits:
            recent_window = int(model_limits["combllama"].get("recent_window_tokens") or 0)
            chunk_size = int(model_limits["combllama"].get("chunk_size") or 0)
            old_tokens = max(0, prompt_tokens - recent_window)
            row["combllama_est_old_tokens"] = old_tokens
            row["combllama_est_num_chunks"] = 0 if chunk_size <= 0 else (old_tokens + chunk_size - 1) // chunk_size
        rows.append(row)

    summary = summarize_rows(rows, thresholds, model_limits, token_counter)
    return rows, summary


def summarize_rows(
    rows: list[dict[str, Any]],
    thresholds: list[int],
    model_limits: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
) -> dict[str, Any]:
    prompt_lengths = [int(row["prompt_tokens"]) for row in rows]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[("benchmark", str(row["benchmark"]))].append(row)
        by_group[("task", f"{row['benchmark']}/{row['task']}")].append(row)
        by_group[("bucket", str(row["length_bucket"]))].append(row)

    groups = []
    for key, group_rows in sorted(by_group.items()):
        lengths = [int(row["prompt_tokens"]) for row in group_rows]
        groups.append(
            {
                "group": "/".join(key),
                "count": len(group_rows),
                "min_prompt_tokens": min(lengths),
                "max_prompt_tokens": max(lengths),
                "avg_prompt_tokens": mean(lengths),
            }
        )

    model_over_limit = {}
    for model_name in model_limits:
        model_over_limit[model_name] = sum(
            1 for row in rows if model_name in str(row["over_limit_models"]).split(",")
        )

    return {
        "record_count": len(rows),
        "tokenizer": token_counter.name,
        "tokenizer_error": token_counter.error,
        "min_prompt_tokens": None if not prompt_lengths else min(prompt_lengths),
        "max_prompt_tokens": None if not prompt_lengths else max(prompt_lengths),
        "avg_prompt_tokens": None if not prompt_lengths else mean(prompt_lengths),
        "threshold_counts": {
            str(threshold): sum(1 for row in rows if row[f"over_{threshold}"])
            for threshold in thresholds
        },
        "model_over_limit_counts": model_over_limit,
        "groups": groups,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Length Audit",
        "",
        f"Tokenizer: `{summary['tokenizer']}`",
        f"Records: {summary['record_count']}",
        f"Prompt tokens: min={summary['min_prompt_tokens']}, avg={summary['avg_prompt_tokens']}, max={summary['max_prompt_tokens']}",
        "",
        "| Threshold | Count Over |",
        "| ---: | ---: |",
    ]
    for threshold, count in summary["threshold_counts"].items():
        lines.append(f"| {threshold} | {count} |")
    lines.extend(["", "| Model | Count Over Limit |", "| --- | ---: |"])
    for model_name, count in summary["model_over_limit_counts"].items():
        lines.append(f"| {model_name} | {count} |")
    lines.extend(["", "| Group | Count | Avg Prompt Tokens | Max Prompt Tokens |", "| --- | ---: | ---: | ---: |"])
    for group in summary["groups"]:
        lines.append(
            f"| {group['group']} | {group['count']} | "
            f"{group['avg_prompt_tokens']:.2f} | {group['max_prompt_tokens']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit benchmark prompt lengths without loading models")
    parser.add_argument("--split", default="dev", choices=["smoke", "dev", "full"])
    parser.add_argument("--benchmarks", nargs="+", default=sorted(BENCHMARKS), choices=sorted(BENCHMARKS))
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--model-configs", nargs="+", default=[
        "benchmarks/configs/models/llama.json",
        "benchmarks/configs/models/combllama.json",
        "benchmarks/configs/models/yoco.json",
        "benchmarks/configs/models/sambay.json",
    ])
    parser.add_argument("--thresholds", nargs="+", type=int, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.split, list(args.benchmarks))
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        for config_path in args.model_configs:
            config = read_json(repo_path(config_path))
            if config.get("tokenizer_path"):
                tokenizer_path = str(config["tokenizer_path"])
                break
    token_counter = TokenCounter(
        tokenizer_path,
        use_chat_template=not args.no_chat_template,
        require_tokenizer=args.require_tokenizer,
    )
    model_limits = load_model_limits(list(args.model_configs))
    rows, summary = audit_examples(
        examples,
        token_counter=token_counter,
        thresholds=list(args.thresholds),
        model_limits=model_limits,
    )
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "benchmarks" / "reports"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    prefix = f"length_audit_{args.split}"
    write_csv(output_dir / f"{prefix}.csv", rows)
    write_json(output_dir / f"{prefix}.json", summary)
    write_markdown(output_dir / f"{prefix}.md", summary)
    print(f"Wrote {len(rows)} rows to {output_dir / f'{prefix}.csv'}")
    if token_counter.error:
        print(f"Tokenizer fallback used because loading failed: {token_counter.error}")


if __name__ == "__main__":
    main()

