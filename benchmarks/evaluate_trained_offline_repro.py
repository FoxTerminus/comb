#!/usr/bin/env python3
"""Run the paper's offline COMB protocol on a locally trained checkpoint.

The prefix-caching arm is reused from the already completed, same-machine
official reproduction.  The LongBench rows and row order are checked exactly
before those baseline measurements are attached to a freshly tokenized input.
The COMB arm then follows ``offline_local_repro.run_comb`` unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from datasets import Dataset

from benchmarks.offline_local_repro import (
    DEFAULT_DATA,
    DEFAULT_TOKENIZER,
    load_local_dataset,
    run_comb,
)
from benchmarks.benchmark_const import CHAT_TEMPLATE_PREFIX
from comb.supported_models import COMB_MODEL_MAPPING
from data import TEST_DATASETS
from data.metrics import qa_f1_score
from training.artifact_io import write_text_atomic


REPO = Path("/data3/junhaohu/comb")
DEFAULT_BASELINE = (
    REPO / "benchmarks/results/offline/meta-llama_Llama-3.1-8B-Instruct"
)
IDENTITY_SOURCES = (
    "benchmarks/evaluate_trained_offline_repro.py",
    "benchmarks/offline_local_repro.py",
    "benchmarks/offline.py",
    "comb/entrypoints/comb.py",
    "comb/integration/vllm/combllama.py",
    "comb/integration/vllm/llm.py",
    "data/LongBench.py",
    "data/metrics.py",
)
BASELINE_COLUMNS = (
    "output_normal",
    "ttft1_normal",
    "ttft2_normal",
    "score_normal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hardware_identity() -> dict:
    physical_gpu = os.environ.get("COMB_BENCHMARK_PHYSICAL_GPU")
    if physical_gpu is None:
        return {"physical_gpu": None, "query_succeeded": False, "name_and_driver": None}
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            physical_gpu,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "physical_gpu": physical_gpu,
        "query_succeeded": result.returncode == 0,
        "name_and_driver": result.stdout.strip() if result.returncode == 0 else None,
    }


def input_identity(args: argparse.Namespace, source: Path, baseline: Path) -> dict:
    parent = Path(args.parent_identity).resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)
    baseline_audit = baseline.parent / "offline_reproduction_audit.json"
    if not baseline_audit.is_file():
        raise FileNotFoundError(baseline_audit)
    return {
        "dataset": args.dataset,
        "examples": args.limit,
        "base_model": args.model,
        "comb_model": str(Path(args.comb_model).resolve()),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "longbench": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        },
        "prefix_baseline": {
            "path": str(baseline.resolve()),
            "sha256": sha256_file(baseline),
            "audit_path": str(baseline_audit.resolve()),
            "audit_sha256": sha256_file(baseline_audit),
        },
        "hardware": hardware_identity(),
        "parent_benchmark_identity": {
            "path": str(parent),
            "sha256": sha256_file(parent),
        },
        "evaluator_sources": {
            relative: sha256_file(REPO / relative) for relative in IDENTITY_SOURCES
        },
    }


def bind_identity(path: Path, identity: dict, result: Path) -> None:
    if path.is_file():
        recorded = json.loads(path.read_text())
        if recorded != identity:
            raise RuntimeError(f"trained offline input identity mismatch: {path}")
        return
    if result.exists():
        raise RuntimeError(f"refusing unbound trained offline result: {result}")
    write_text_atomic(path, json.dumps(identity, indent=2) + "\n")


def attach_prefix_baseline(data: Dataset, baseline: Dataset) -> Dataset:
    if len(data) == 0 or len(data) != len(baseline):
        raise RuntimeError(
            f"input/baseline row count mismatch: input={len(data)} baseline={len(baseline)}"
        )
    missing = sorted(set(BASELINE_COLUMNS) - set(baseline.column_names))
    if missing:
        raise RuntimeError(f"prefix baseline is missing columns: {missing}")
    for key in ("input", "context", "answers"):
        if data[key] != baseline[key]:
            raise RuntimeError(f"prefix baseline row identity mismatch: {key}")
    for key in BASELINE_COLUMNS:
        data = data.add_column(key, baseline[key])
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(TEST_DATASETS), required=True)
    parser.add_argument("--comb-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-identity", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.data_dir / f"{args.dataset}.jsonl"
    baseline_path = args.baseline_dir / f"{args.dataset}.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(args.tokenizer)
    if not 1 <= args.limit <= 200:
        raise ValueError("--limit must be in [1, 200]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.dataset}.json"
    identity_path = args.output_dir / f"{args.dataset}_input_manifest.json"
    identity = input_identity(args, source, baseline_path)
    bind_identity(identity_path, identity, result_path)
    if result_path.is_file() and result_path.stat().st_size:
        print(f"Result exists and identity matches: {result_path}", flush=True)
        return

    os.environ["COMB_TOKENIZER_PATH"] = str(args.tokenizer.resolve())
    dataset_cls = TEST_DATASETS[args.dataset]
    dataset = load_local_dataset(args.model, dataset_cls, source)
    dataset.data = dataset.data.select(range(args.limit))
    baseline = Dataset.from_json(str(baseline_path)).select(range(args.limit))
    data = attach_prefix_baseline(dataset.data, baseline)

    COMB_MODEL_MAPPING[args.model] = str(Path(args.comb_model).resolve())
    max_len = 128 if dataset_cls.metric == qa_f1_score else 4096
    data = run_comb(
        args.model,
        data,
        CHAT_TEMPLATE_PREFIX[args.model],
        max_len,
    )
    data = data.map(dataset_cls.scorer, fn_kwargs={"method": "comb"})
    result = data.remove_columns(
        [
            "normal_input",
            "input_ids",
            "normal_input_new",
            "input_ids_new",
            "chunk_ids",
            "cross_attention_mask",
            "labels",
        ]
    )
    temporary = result_path.with_name(f".{result_path.name}.tmp-{os.getpid()}")
    result.to_json(temporary)
    os.replace(temporary, result_path)
    print(f"Published {result_path}", flush=True)


if __name__ == "__main__":
    main()
