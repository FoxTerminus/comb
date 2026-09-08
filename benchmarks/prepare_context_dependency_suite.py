#!/usr/bin/env python3
"""Build a deterministic, held-out PIC context-dependency evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from datasets import Dataset, load_dataset
import numpy as np
from transformers import AutoTokenizer


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
QA_INSTRUCTION = (
    "Answer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\nQuestion: "
)
X_SUM_INSTRUCTION = (
    "You are an AI assistant. Read the provided text and produce a concise "
    "summary. Capture the main points without unnecessary details."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    parser.add_argument("--xsum-validation-parquet")
    parser.add_argument("--longbench-zip")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_one(root: Path, pattern: str, description: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"cannot find cached {description} below {root}")
    return matches[-1].resolve()


def tokenize_row(
    tokenizer,
    *,
    query: str,
    context: str,
    answer: str,
    source_dataset: str,
    source_split: str,
    source_index: int,
    source_id: str,
) -> dict | None:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}]
    )
    context_tokens = tokenizer(context)
    labels = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": answer}]
    )
    token_count = len(context_tokens["input_ids"])
    # Mirror the model contract without silently modifying held-out examples.
    if not (0 < token_count <= 16_384):
        return None
    if len(input_ids) - 1 >= 512 or len(input_ids) + len(labels) <= 1:
        return None
    return {
        "input_ids": input_ids,
        "chunk_ids": context_tokens["input_ids"],
        "cross_attention_mask": context_tokens["attention_mask"],
        "labels": labels,
        "token_count": token_count,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_index": source_index,
        "source_id": source_id,
    }


def select_evenly(
    rows: list[dict], count: int, *, minimum_exclusive: int, maximum: int
) -> list[dict]:
    eligible = [
        row
        for row in rows
        if minimum_exclusive < row["token_count"] <= maximum
    ]
    if len(eligible) < count:
        raise ValueError(
            f"need {count} rows in ({minimum_exclusive}, {maximum}], "
            f"found {len(eligible)}"
        )
    indices = np.linspace(0, len(eligible) - 1, num=count, dtype=np.int64)
    return [eligible[int(index)] for index in indices]


def unique_contexts(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        signature = hashlib.sha256(
            np.asarray(row["chunk_ids"], dtype=np.int32).tobytes()
        ).digest()
        if signature not in seen:
            seen.add(signature)
            unique.append(row)
    return unique


def process_squad(tokenizer) -> tuple[list[dict], str]:
    raw = load_dataset("squad_v2", split="validation")
    rows = []
    for index, example in enumerate(raw):
        answers = example["answers"]["text"]
        if not answers:
            continue
        row = tokenize_row(
            tokenizer,
            query=example["question"],
            context=example["context"],
            answer=answers[0],
            source_dataset="squad_v2",
            source_split="validation",
            source_index=index,
            source_id=example["id"],
        )
        if row is not None:
            rows.append(row)
    return unique_contexts(rows), str(raw._fingerprint)


def process_xsum(tokenizer, parquet_path: Path) -> list[dict]:
    raw = Dataset.from_parquet(str(parquet_path))
    rows = []
    for index, example in enumerate(raw):
        row = tokenize_row(
            tokenizer,
            query=X_SUM_INSTRUCTION,
            context=example["document"],
            answer=example["summary"],
            source_dataset="xsum",
            source_split="validation",
            source_index=index,
            source_id=example["id"],
        )
        if row is not None:
            rows.append(row)
    return unique_contexts(rows)


def process_longbench(tokenizer, archive_path: Path) -> dict[str, list[dict]]:
    result = {}
    with zipfile.ZipFile(archive_path) as archive:
        for task in ("qasper", "2wikimqa", "hotpotqa"):
            rows = []
            payload = archive.read(f"data/{task}.jsonl")
            for index, line in enumerate(payload.splitlines()):
                example = json.loads(line)
                if not example["answers"]:
                    continue
                row = tokenize_row(
                    tokenizer,
                    query=QA_INSTRUCTION + example["input"],
                    context=example["context"],
                    answer=example["answers"][0],
                    source_dataset=f"longbench_v1/{task}",
                    source_split="test-as-development",
                    source_index=index,
                    source_id=str(example.get("_id", index)),
                )
                if row is not None:
                    rows.append(row)
            result[task] = unique_contexts(rows)
    return result


def save_panel(root: Path, name: str, rows: list[dict]) -> dict:
    if len(rows) < 2:
        raise ValueError(f"panel {name} requires at least two rows")
    panel_path = root / name
    Dataset.from_list(rows).save_to_disk(str(panel_path))
    maximum_tokens = max(row["token_count"] for row in rows)
    return {
        "name": name,
        "path": str(panel_path),
        "target_column": "labels",
        "examples": len(rows),
        "batch_size": 4 if maximum_tokens <= 4_096 else 1,
        "context_token_range": {
            "minimum": min(row["token_count"] for row in rows),
            "maximum": maximum_tokens,
            "mean": sum(row["token_count"] for row in rows) / len(rows),
        },
        "source_counts": {
            source: sum(row["source_dataset"] == source for row in rows)
            for source in sorted({row["source_dataset"] for row in rows})
        },
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing suite: {output_root}")
    hf_home = Path(args.hf_home or Path.home() / ".cache/huggingface").resolve()
    tokenizer_path = (
        Path(args.tokenizer_path).resolve()
        if args.tokenizer_path
        else discover_one(
            hf_home,
            "hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*",
            "Llama tokenizer snapshot",
        )
    )
    xsum_path = (
        Path(args.xsum_validation_parquet).resolve()
        if args.xsum_validation_parquet
        else discover_one(
            hf_home,
            "hub/datasets--EdinburghNLP--xsum/snapshots/*/default/validation/0000.parquet",
            "XSum validation parquet",
        )
    )
    longbench_path = (
        Path(args.longbench_zip).resolve()
        if args.longbench_zip
        else discover_one(
            hf_home,
            "hub/datasets--THUDM--LongBench/snapshots/*/data.zip",
            "LongBench archive",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    squad, squad_fingerprint = process_squad(tokenizer)
    xsum = process_xsum(tokenizer, xsum_path)
    longbench = process_longbench(tokenizer, longbench_path)

    panels = {
        "squad_v2_validation_short": select_evenly(
            squad, 64, minimum_exclusive=0, maximum=2_048
        ),
        "xsum_validation_short": select_evenly(
            xsum, 64, minimum_exclusive=0, maximum=2_048
        ),
        "xsum_validation_medium": select_evenly(
            xsum, 32, minimum_exclusive=2_048, maximum=8_192
        ),
        "qasper_medium": select_evenly(
            longbench["qasper"], 64, minimum_exclusive=2_048, maximum=8_192
        ),
        "multidoc_medium": (
            select_evenly(
                longbench["2wikimqa"], 36,
                minimum_exclusive=2_048, maximum=8_192,
            )
            + select_evenly(
                longbench["hotpotqa"], 28,
                minimum_exclusive=2_048, maximum=8_192,
            )
        ),
        "multidoc_long": (
            select_evenly(
                longbench["2wikimqa"], 32,
                minimum_exclusive=8_192, maximum=16_384,
            )
            + select_evenly(
                longbench["hotpotqa"], 32,
                minimum_exclusive=8_192, maximum=16_384,
            )
        ),
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent)
    )
    try:
        entries = [save_panel(staging, name, rows) for name, rows in panels.items()]
        for entry in entries:
            entry["path"] = str(output_root / entry["name"])
        manifest = {
            "version": 1,
            "name": "frozen32_pic_heldout_v1",
            "description": (
                "Fixed held-out PIC panels; LongBench v1 is development-only and "
                "LongBench v2 remains sealed for final evaluation."
            ),
            "model_tokenizer": MODEL_NAME,
            "tokenizer_path": str(tokenizer_path),
            "sources": {
                "squad_v2_validation_fingerprint": squad_fingerprint,
                "xsum_validation_parquet": str(xsum_path),
                "xsum_validation_sha256": sha256_file(xsum_path),
                "longbench_v1_archive": str(longbench_path),
                "longbench_v1_sha256": sha256_file(longbench_path),
            },
            "datasets": entries,
        }
        (staging / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        os.replace(staging, output_root)
    except BaseException:
        # Leave the staging directory for diagnosis; never replace a valid suite.
        raise
    print(output_root / "suite_manifest.json")


if __name__ == "__main__":
    main()
