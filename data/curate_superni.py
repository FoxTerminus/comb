#!/usr/bin/env python3
"""Create a deterministic, length-balanced Super-NI training subset."""

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

from data.SuperNI import split_superni_prompt


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SOURCE_NAME = "Super-Natural-Instructions"
OUTPUT_NAME = "Super-Natural-Instructions-Curated"
TEACHER_COLUMN = MODEL_NAME
KEEP_COLUMNS = [
    "prompt",
    "response",
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


def exact_sample_size(count: int, fraction: float) -> int:
    if count <= 0 or fraction <= 0:
        return 0
    return min(count, max(1, int(math.floor(count * fraction + 0.5))))


def task_seed(task: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bucket_counts(values: np.ndarray) -> dict[str, int]:
    edges = (0, 256, 512, 1024, 2048)
    return {
        f"{low + 1}-{high}": int(((values > low) & (values <= high)).sum())
        for low, high in zip(edges[:-1], edges[1:])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=cache_path(SOURCE_NAME))
    parser.add_argument("--output", type=Path, default=cache_path(OUTPUT_NAME))
    parser.add_argument("--short-threshold", type=int, default=256)
    parser.add_argument("--short-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-context-tokens", type=int, default=2048)
    parser.add_argument("--maximum-decoder-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-free-gib", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 < args.short_fraction <= 1:
        raise ValueError("--short-fraction must be in (0, 1]")
    if args.short_threshold <= 0:
        raise ValueError("--short-threshold must be positive")
    if args.maximum_context_tokens < args.short_threshold:
        raise ValueError("maximum context must be at least the short threshold")
    if args.maximum_decoder_tokens <= 0:
        raise ValueError("--maximum-decoder-tokens must be positive")

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
    context_valid = (context_lengths > 0) & (
        context_lengths <= args.maximum_context_tokens
    )
    target_fits = instruction_lengths + teacher_lengths <= (
        args.maximum_decoder_tokens + 1
    )
    eligible = context_valid & target_fits
    short_eligible = eligible & (context_lengths <= args.short_threshold)
    longer_eligible = eligible & (context_lengths > args.short_threshold)

    short_by_task: dict[str, list[int]] = defaultdict(list)
    offset = 0
    prompts = dataset.select_columns(["prompt"])
    for batch in prompts.iter(batch_size=50_000):
        for relative_index, prompt in enumerate(batch["prompt"]):
            index = offset + relative_index
            if short_eligible[index]:
                instruction, _ = split_superni_prompt(prompt)
                short_by_task[instruction].append(index)
        offset += len(batch["prompt"])
    if offset != len(dataset):
        raise RuntimeError(f"prompt scan stopped early: {offset} != {len(dataset)}")

    selected_short: list[int] = []
    task_records = []
    for task, indices in short_by_task.items():
        count = exact_sample_size(len(indices), args.short_fraction)
        rng = random.Random(task_seed(task, args.seed))
        chosen = rng.sample(indices, count)
        selected_short.extend(chosen)
        task_records.append(
            {
                "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
                "eligible_short_rows": len(indices),
                "selected_short_rows": count,
            }
        )

    selected_longer = np.flatnonzero(longer_eligible).tolist()
    selected_indices = sorted(selected_short + selected_longer)
    selected_array = np.asarray(selected_indices, dtype="<i8")
    selected_lengths = context_lengths[selected_array]
    index_sha256 = hashlib.sha256(selected_array.tobytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": str(source_path),
        "source_dataset_fingerprint": str(dataset._fingerprint),
        "source_rows": len(dataset),
        "teacher_column": TEACHER_COLUMN,
        "criteria": {
            "short_threshold": args.short_threshold,
            "short_fraction_per_task": args.short_fraction,
            "maximum_context_tokens": args.maximum_context_tokens,
            "maximum_decoder_tokens": args.maximum_decoder_tokens,
            "require_complete_teacher_target": True,
            "seed": args.seed,
        },
        "audit": {
            "source_context_buckets": bucket_counts(context_lengths),
            "invalid_context_rows": int((~context_valid).sum()),
            "teacher_targets_at_512_tokens": int((teacher_lengths == 512).sum()),
            "teacher_targets_that_would_be_truncated": int((~target_fits).sum()),
            "fully_masked_rows": int(
                (instruction_lengths > args.maximum_decoder_tokens).sum()
            ),
            "eligible_short_rows": int(short_eligible.sum()),
            "selected_short_rows": len(selected_short),
            "eligible_longer_rows": int(longer_eligible.sum()),
            "selected_rows": len(selected_indices),
            "selected_context_buckets": bucket_counts(selected_lengths),
            "stratification_tasks": len(short_by_task),
            "selected_indices_sha256": index_sha256,
        },
        "task_records": sorted(
            task_records,
            key=lambda item: (-item["eligible_short_rows"], item["task_sha256"]),
        ),
        "output": str(output_path),
    }
    print(json.dumps({key: value for key, value in manifest.items() if key != "task_records"}, indent=2))
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
    print(json.dumps({"saved": str(output_path), "rows": len(reloaded)}))


if __name__ == "__main__":
    main()
