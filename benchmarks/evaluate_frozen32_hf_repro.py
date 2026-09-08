#!/usr/bin/env python3
"""Resumable LongBench evaluation for an exported Frozen32 PIC checkpoint.

The row order, prompts, corrected answers, and metrics match
``evaluate_hf_repro.py``.  Context is encoded once per example and converted
to canonical PIC K/V tensors before generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from benchmarks.evaluate_hf_repro import (
    DEFAULT_DATA,
    DEFAULT_TOKENIZER,
    PAPER_TARGETS,
    existing_indices,
    load_rows,
)
from comb.integration.hf.CombLlamaFrozen32 import (
    CombLlamaFrozen32ForConditionalGeneration,
)
from data import TEST_DATASETS
from data.metrics import qa_f1_score
from training.artifact_io import append_jsonl_fsync, read_resumable_jsonl, write_text_atomic


DEFAULT_MODEL = (
    "/data3/junhaohu/checkpoints/CombLlamaFrozen32/"
    "hf_export_step_00100000/hf_step_00100000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=",".join(TEST_DATASETS))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--qa-max-new-tokens", type=int, default=128)
    parser.add_argument("--summary-max-new-tokens", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--rope-buffer-dtype", choices=("fp32", "bf16"), default="fp32"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_manifest(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> None:
    model = Path(args.model).resolve()
    data = Path(args.data_dir).resolve()
    identity_files = [model / "config.json"]
    index = model / "model.safetensors.index.json"
    if index.is_file():
        identity_files.append(index)
    manifest = {
        "model": str(model),
        "model_identity": {str(path): sha256_file(path) for path in identity_files},
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "data_dir": str(data),
        "data_identity": {
            name: sha256_file(data / f"{name}.jsonl") for name in datasets
        },
        "datasets": datasets,
        "start_index": args.start_index,
        "limit": args.limit,
        "qa_max_new_tokens": args.qa_max_new_tokens,
        "summary_max_new_tokens": args.summary_max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "rope_buffer_dtype": args.rope_buffer_dtype,
        "pic_construction": "build_pic_cache_once_per_example",
        "transformers": transformers.__version__,
    }
    path = output_dir / "benchmark_input_manifest.json"
    text = json.dumps(manifest, indent=2) + "\n"
    if path.is_file():
        if path.read_text() != text:
            raise RuntimeError(f"benchmark input identity mismatch: {path}")
    else:
        write_text_atomic(path, text)


def main() -> None:
    args = parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets).difference(TEST_DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    if not 0 <= args.start_index < args.limit <= 200:
        raise ValueError("require 0 <= start-index < limit <= 200")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bind_manifest(args, output_dir, datasets)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = CombLlamaFrozen32ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    rope_dtype = torch.float32 if args.rope_buffer_dtype == "fp32" else torch.bfloat16
    rope_buffers = []
    for name, module in model.named_modules():
        if "inv_freq" in module._buffers and module._buffers["inv_freq"] is not None:
            module._buffers["inv_freq"] = module._buffers["inv_freq"].to(rope_dtype)
            if getattr(module, "original_inv_freq", None) is not None:
                module.original_inv_freq = module._buffers["inv_freq"]
            rope_buffers.append(f"{name}.inv_freq")
    if not rope_buffers:
        raise RuntimeError("No RoPE inv_freq buffers found")
    print(json.dumps({"rope_buffers": rope_buffers}), flush=True)

    for dataset_name in datasets:
        dataset_cls = TEST_DATASETS[dataset_name]
        rows = load_rows(Path(args.data_dir) / f"{dataset_name}.jsonl", args.limit)
        result_path = output_dir / f"{dataset_name}.jsonl"
        finished = existing_indices(result_path)
        max_new_tokens = (
            args.qa_max_new_tokens
            if dataset_cls.metric == qa_f1_score
            else args.summary_max_new_tokens
        )
        for index, row in enumerate(rows[args.start_index :], start=args.start_index):
            if index in finished:
                continue
            answers = row["answers"]
            if hasattr(dataset_cls, "correct_answers"):
                answers = dataset_cls.correct_answers.get(row["input"], answers)
            question_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": dataset_cls.instruction + row["input"]}],
                return_tensors="pt",
            ).cuda()
            chunk = tokenizer(row["context"], return_tensors="pt").to("cuda")
            started = time.perf_counter()
            with torch.inference_mode():
                pic_cache = model.build_pic_cache(chunk.input_ids, chunk.attention_mask)
                output = model.generate(
                    input_ids=question_ids,
                    attention_mask=torch.ones_like(question_ids),
                    cross_attention_states=pic_cache,
                    cross_attention_mask=chunk.attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            prediction = tokenizer.decode(
                output[0, question_ids.shape[-1] :], skip_special_tokens=True
            ).replace("assistant\n\n", "")
            score_prediction = prediction
            if dataset_name == "samsum":
                score_prediction = score_prediction.lstrip("\n").split("\n")[0]
            score = max(dataset_cls.metric(score_prediction, answer) for answer in answers)
            record = {
                "index": index,
                "dataset": dataset_name,
                "input": row["input"],
                "answers": answers,
                "prediction": prediction,
                "score": score,
                "question_tokens": int(question_ids.shape[-1]),
                "context_tokens": int(chunk.input_ids.shape[-1]),
                "generated_tokens": int(output.shape[-1] - question_ids.shape[-1]),
                "seconds": time.perf_counter() - started,
                "backend": f"frozen32-hf-transformers-{transformers.__version__}",
                "rope_buffer_dtype": args.rope_buffer_dtype,
                "pic_construction": "build_pic_cache_once_per_example",
            }
            append_jsonl_fsync(result_path, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del pic_cache, output, chunk, question_ids

        completed = [
            row
            for row in read_resumable_jsonl(result_path)
            if args.start_index <= int(row["index"]) < args.limit
        ]
        if len(completed) != args.limit - args.start_index:
            raise RuntimeError(
                f"incomplete result for {dataset_name}: {len(completed)} rows"
            )
        mean_score = sum(float(row["score"]) for row in completed) / len(completed)
        summary = {
            "dataset": dataset_name,
            "completed": len(completed),
            "mean_score": mean_score,
            "paper_target": PAPER_TARGETS[dataset_name],
            "delta_from_paper": mean_score - PAPER_TARGETS[dataset_name],
            "rope_buffer_dtype": args.rope_buffer_dtype,
            "pic_construction": "build_pic_cache_once_per_example",
        }
        write_text_atomic(
            output_dir / f"{dataset_name}_summary.json",
            json.dumps(summary, indent=2) + "\n",
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
