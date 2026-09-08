#!/usr/bin/env python3
"""Run the official offline experiment from already downloaded LongBench JSONL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from datasets import Dataset
from transformers import AutoTokenizer

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from benchmark_const import CHAT_TEMPLATE_PREFIX
from comb import COMB
from data import TEST_DATASETS
from data.metrics import qa_f1_score
from offline import run_vllm


DEFAULT_DATA = Path(
    "/data3/junhaohu/.cache/huggingface/datasets/downloads/extracted/"
    "6580f247ded331d33b7e904e0824dacdb4db596a48098cd32cb7136084ce8bd3/data"
)

DEFAULT_TOKENIZER = Path(
    "/data3/junhaohu/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)


def run_comb(model_name, data, template_prefix, max_len):
    """Reuse the official COMB loop while loading the cached tokenizer locally."""
    comb = COMB(model_name, disable_log_stats=False, pbc_memory_utilization=0.5)
    comb.set_sampling_params(temperature=0.0, max_tokens=max_len)
    tokenizer_path = Path(os.environ.get("COMB_TOKENIZER_PATH", DEFAULT_TOKENIZER))
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"local tokenizer snapshot not found: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    answer, ttft1, ttft2 = [], [], []
    for prompt in data:
        output = comb.generate(prompt)[0]
        answer.append(
            tokenizer.decode(output.token_ids, skip_special_tokens=True).replace(
                template_prefix, ""
            )
        )
        ttft1.append(output.first_token_latency)
        prompt["input_ids"] = prompt.pop("input_ids_new")
        output = comb.generate(prompt, max_tokens=1)[0]
        ttft2.append(output.first_token_latency)

    data = data.add_column("output_comb", answer)
    data = data.add_column("ttft1_comb", ttft1)
    data = data.add_column("ttft2_comb", ttft2)
    return data


def load_local_dataset(model_name: str, dataset_cls, path: Path):
    """Bypass only the remote HF builder; reuse the official row transforms."""
    dataset = object.__new__(dataset_cls)
    dataset.model_name = model_name
    dataset.max_input_length = 512
    dataset._init_tokenizer()
    data = Dataset.from_json(str(path))
    if hasattr(dataset_cls, "correct_answers"):
        data = data.map(dataset._correct)
    dataset.data = data.map(dataset._prepare_input)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=sorted(TEST_DATASETS), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    dataset_cls = TEST_DATASETS[args.dataset]
    source = args.data_dir / f"{args.dataset}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    dataset = load_local_dataset(args.model, dataset_cls, source)
    if len(dataset) != 200:
        raise RuntimeError(f"expected 200 LongBench rows, found {len(dataset)}")

    result_dir = Path("results/offline") / args.model.replace("/", "_")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{args.dataset}.json"
    normal_checkpoint = result_dir / f".{args.dataset}.normal.json"
    if result_path.exists():
        print(f"Result exists: {result_path}", flush=True)
        return

    max_len = 128 if dataset_cls.metric == qa_f1_score else 4096
    template_prefix = CHAT_TEMPLATE_PREFIX[args.model]
    if normal_checkpoint.is_file() and normal_checkpoint.stat().st_size:
        print(f"Loading completed Prefix Caching phase: {normal_checkpoint}", flush=True)
        data = Dataset.from_json(str(normal_checkpoint))
        required = {"output_normal", "ttft1_normal", "ttft2_normal", "score_normal"}
        if len(data) != 200 or not required.issubset(data.column_names):
            raise RuntimeError(f"invalid Prefix Caching checkpoint: {normal_checkpoint}")
    else:
        print(f"Starting Prefix Caching phase for {args.dataset}", flush=True)
        data = run_vllm(args.model, dataset.data, template_prefix, max_len)
        data = data.map(dataset_cls.scorer, fn_kwargs={"method": "normal"})
        temporary_normal = normal_checkpoint.with_name(
            f".{normal_checkpoint.name}.tmp-{os.getpid()}"
        )
        data.to_json(temporary_normal)
        os.replace(temporary_normal, normal_checkpoint)
        print(f"Completed Prefix Caching phase: {normal_checkpoint}", flush=True)

    print(f"Starting COMB phase for {args.dataset}", flush=True)
    data = run_comb(args.model, data, template_prefix, max_len)
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
    normal_checkpoint.unlink(missing_ok=True)
    print(f"Published {result_path}", flush=True)


if __name__ == "__main__":
    main()
