#!/usr/bin/env python3
"""Create a deterministic, task-balanced Natural-Instructions subset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

from datasets import load_from_disk
import numpy as np
import pyarrow.compute as pc


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SOURCE_NAME = "Natural-Instructions"
OUTPUT_NAME = "Natural-Instructions-Curated"
TEACHER_COLUMN = MODEL_NAME
EOS_TOKEN_IDS = (128001, 128008, 128009)
DEFAULT_EXCLUDED_TASK_SUBSTRINGS = (
    "hotpotqa",
    "multi_news",
    "multinews",
    "samsum",
    "musique",
    "2wikimqa",
    "xsum",
)
KEEP_COLUMNS = [
    "task_name",
    "id",
    "definition",
    "inputs",
    "targets",
    "normal_input",
    "input_ids",
    "chunk_ids",
    "cross_attention_mask",
    "labels",
    "token_count",
    TEACHER_COLUMN,
]


def cache_path(name: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    safe_model = MODEL_NAME.replace("/", "_")
    return hf_home / "datasets" / f"{name}_{safe_model}"


def exact_sample_size(count: int, fraction: float, cap: int) -> int:
    if count <= 0 or fraction <= 0 or cap <= 0:
        return 0
    requested = max(1, int(math.floor(count * fraction + 0.5)))
    return min(count, requested, cap)


def task_seed(task: str, stratum: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{stratum}\0{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_sample(
    indices: list[int], *, count: int, task: str, stratum: str, seed: int
) -> list[int]:
    if count >= len(indices):
        return list(indices)
    return random.Random(task_seed(task, stratum, seed)).sample(indices, count)


def bucket_counts(values: np.ndarray) -> dict[str, int]:
    edges = (0, 256, 512, 1024, 2048, 4096, 8192, 16384)
    result = {
        f"{low + 1}-{high}": int(((values > low) & (values <= high)).sum())
        for low, high in zip(edges[:-1], edges[1:])
    }
    result["outside"] = int(((values <= 0) | (values > edges[-1])).sum())
    return result


def final_tokens(chunked_array) -> np.ndarray:
    """Return each list row's final token without materializing Python lists."""
    parts = []
    for chunk in chunked_array.chunks:
        offsets = np.asarray(chunk.offsets)
        lengths = offsets[1:] - offsets[:-1]
        values = np.asarray(chunk.values)
        result = np.full(len(lengths), -1, dtype=np.int64)
        nonempty = lengths > 0
        result[nonempty] = values[offsets[1:][nonempty] - 1]
        parts.append(result)
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=cache_path(SOURCE_NAME))
    parser.add_argument("--output", type=Path, default=cache_path(OUTPUT_NAME))
    parser.add_argument("--short-fraction", type=float, default=0.10)
    parser.add_argument("--short-cap-per-task", type=int, default=2048)
    parser.add_argument("--medium-fraction", type=float, default=0.50)
    parser.add_argument("--medium-cap-per-task", type=int, default=2048)
    parser.add_argument("--long-cap-per-task", type=int, default=8000)
    parser.add_argument("--maximum-context-tokens", type=int, default=16384)
    parser.add_argument("--maximum-decoder-tokens", type=int, default=512)
    parser.add_argument(
        "--excluded-task-substrings",
        default=",".join(DEFAULT_EXCLUDED_TASK_SUBSTRINGS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name, value in (
        ("short fraction", args.short_fraction),
        ("medium fraction", args.medium_fraction),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if min(
        args.short_cap_per_task,
        args.medium_cap_per_task,
        args.long_cap_per_task,
        args.maximum_context_tokens,
        args.maximum_decoder_tokens,
    ) <= 0:
        raise ValueError("caps and maximum lengths must be positive")

    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if shutil.disk_usage(output_path.parent).free / 1024**3 < args.minimum_free_gib:
        raise RuntimeError("free disk is below --minimum-free-gib")

    dataset = load_from_disk(source_path)
    missing = sorted(set(KEEP_COLUMNS).difference(dataset.column_names))
    if missing:
        raise RuntimeError(f"source cache is missing required columns: {missing}")

    table = dataset._data.table
    context_lengths = np.asarray(table["token_count"].combine_chunks())
    instruction_lengths = np.asarray(
        pc.list_value_length(table["input_ids"]).combine_chunks()
    )
    teacher_lengths = np.asarray(
        pc.list_value_length(table[TEACHER_COLUMN]).combine_chunks()
    )
    teacher_final_tokens = final_tokens(table[TEACHER_COLUMN])
    ends_with_eos = np.isin(teacher_final_tokens, EOS_TOKEN_IDS)
    context_valid = (context_lengths > 0) & (
        context_lengths <= args.maximum_context_tokens
    )
    target_fits = instruction_lengths + teacher_lengths <= (
        args.maximum_decoder_tokens + 1
    )
    eligible = context_valid & target_fits & ends_with_eos

    excluded_substrings = tuple(
        value.strip().lower()
        for value in args.excluded_task_substrings.split(",")
        if value.strip()
    )
    strata: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"short": [], "medium": [], "long": []}
    )
    excluded_task_rows = 0
    eligible_before_task_exclusion = int(eligible.sum())
    offset = 0
    for batch in dataset.select_columns(["task_name"]).iter(batch_size=100_000):
        for relative_index, task in enumerate(batch["task_name"]):
            index = offset + relative_index
            if not eligible[index]:
                continue
            normalized = task.lower()
            if any(value in normalized for value in excluded_substrings):
                excluded_task_rows += 1
                continue
            length = int(context_lengths[index])
            stratum = "short" if length <= 256 else "medium" if length <= 512 else "long"
            strata[task][stratum].append(index)
        offset += len(batch["task_name"])
    if offset != len(dataset):
        raise RuntimeError(f"task scan stopped early: {offset} != {len(dataset)}")

    selected_indices: list[int] = []
    task_records = []
    for task, groups in strata.items():
        short_count = exact_sample_size(
            len(groups["short"]), args.short_fraction, args.short_cap_per_task
        )
        medium_count = exact_sample_size(
            len(groups["medium"]), args.medium_fraction, args.medium_cap_per_task
        )
        long_count = min(len(groups["long"]), args.long_cap_per_task)
        chosen = {
            "short": deterministic_sample(
                groups["short"], count=short_count, task=task, stratum="short", seed=args.seed
            ),
            "medium": deterministic_sample(
                groups["medium"], count=medium_count, task=task, stratum="medium", seed=args.seed
            ),
            "long": deterministic_sample(
                groups["long"], count=long_count, task=task, stratum="long", seed=args.seed
            ),
        }
        selected_indices.extend(chosen["short"] + chosen["medium"] + chosen["long"])
        task_records.append(
            {
                "task_name": task,
                "source": {name: len(values) for name, values in groups.items()},
                "selected": {name: len(values) for name, values in chosen.items()},
            }
        )

    selected_indices.sort()
    selected_array = np.asarray(selected_indices, dtype="<i8")
    selected_lengths = context_lengths[selected_array]
    manifest = {
        "schema_version": 1,
        "source": str(source_path),
        "source_dataset_fingerprint": str(dataset._fingerprint),
        "source_rows": len(dataset),
        "teacher_column": TEACHER_COLUMN,
        "criteria": {
            "require_teacher_eos": True,
            "eos_token_ids": list(EOS_TOKEN_IDS),
            "require_complete_teacher_target": True,
            "maximum_decoder_tokens": args.maximum_decoder_tokens,
            "maximum_context_tokens": args.maximum_context_tokens,
            "short_range": [1, 256],
            "short_fraction_per_task": args.short_fraction,
            "short_cap_per_task": args.short_cap_per_task,
            "medium_range": [257, 512],
            "medium_fraction_per_task": args.medium_fraction,
            "medium_cap_per_task": args.medium_cap_per_task,
            "long_range": [513, args.maximum_context_tokens],
            "long_cap_per_task": args.long_cap_per_task,
            "excluded_task_substrings": list(excluded_substrings),
            "seed": args.seed,
        },
        "audit": {
            "source_context_buckets": bucket_counts(context_lengths),
            "invalid_context_rows": int((~context_valid).sum()),
            "teacher_targets_without_terminal_eos": int((~ends_with_eos).sum()),
            "teacher_targets_that_would_be_truncated": int((~target_fits).sum()),
            "eligible_before_task_exclusion": eligible_before_task_exclusion,
            "excluded_evaluation_task_rows": excluded_task_rows,
            "eligible_tasks": len(strata),
            "selected_rows": len(selected_indices),
            "selected_context_buckets": bucket_counts(selected_lengths),
            "selected_indices_sha256": hashlib.sha256(selected_array.tobytes()).hexdigest(),
        },
        "task_records": sorted(task_records, key=lambda item: item["task_name"]),
        "output": str(output_path),
    }
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "task_records"},
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp-{os.getpid()}")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_path}")
    curated = dataset.select(selected_indices).select_columns(KEEP_COLUMNS)
    curated.save_to_disk(temporary_path)
    reloaded = load_from_disk(temporary_path)
    if len(reloaded) != len(selected_indices):
        raise RuntimeError("curated dataset row count changed after serialization")
    manifest["output_dataset_fingerprint"] = str(reloaded._fingerprint)
    manifest["output_bytes"] = sum(
        path.stat().st_size for path in temporary_path.rglob("*") if path.is_file()
    )
    (temporary_path / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    temporary_path.replace(output_path)
    print(json.dumps({"saved": str(output_path), "rows": len(reloaded)}), flush=True)


if __name__ == "__main__":
    main()
