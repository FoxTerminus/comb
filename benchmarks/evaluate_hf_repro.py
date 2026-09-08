"""Resumable accuracy evaluation of an official Comb checkpoint on LongBench.

This is an additive reproduction utility.  It keeps the prompts and scorers from
``data/LongBench.py`` but reads the already downloaded JSONL files directly, so
the evaluation is deterministic and does not require network access.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from data import TEST_DATASETS
from data.metrics import qa_f1_score
from training.artifact_io import (
    append_jsonl_fsync,
    read_resumable_jsonl,
    write_text_atomic,
)


DEFAULT_MODEL = "/data3/junhaohu/model/official_CombLlama-11B-Instruct"
DEFAULT_TOKENIZER = (
    "/data3/junhaohu/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_DATA = (
    "/data3/junhaohu/.cache/huggingface/datasets/downloads/extracted/"
    "6580f247ded331d33b7e904e0824dacdb4db596a48098cd32cb7136084ce8bd3/data"
)
PAPER_TARGETS = {
    "hotpotqa": 0.48,
    "musique": 0.31,
    "2wikimqa": 0.40,
    "multi_news": 0.22,
    "samsum": 0.07,
}


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
        "--rope-buffer-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Precision of non-persistent RoPE inv_freq buffers after loading.",
    )
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def existing_indices(path: Path) -> set[int]:
    records = read_resumable_jsonl(path)
    indices = [int(record["index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"duplicate result index in {path}")
    return set(indices)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = CombLlamaForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
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
    print(
        json.dumps(
            {
                "rope_buffer_dtype": args.rope_buffer_dtype,
                "rope_buffers": rope_buffers,
            }
        ),
        flush=True,
    )

    for dataset_name in args.datasets.split(","):
        dataset_name = dataset_name.strip()
        dataset_cls = TEST_DATASETS[dataset_name]
        rows = load_rows(Path(args.data_dir) / f"{dataset_name}.jsonl", args.limit)
        result_path = output_dir / f"{dataset_name}.jsonl"
        finished = existing_indices(result_path)
        max_new_tokens = (
            args.qa_max_new_tokens
            if dataset_cls.metric == qa_f1_score
            else args.summary_max_new_tokens
        )
        for index, row in enumerate(rows[args.start_index:], start=args.start_index):
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
                output = model.generate(
                    input_ids=question_ids,
                    attention_mask=torch.ones_like(question_ids),
                    chunk_ids=chunk.input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            prediction = tokenizer.decode(
                output[0, question_ids.shape[-1]:], skip_special_tokens=True
            ).replace("assistant\n\n", "")
            score_prediction = prediction
            if dataset_name == "samsum":
                score_prediction = score_prediction.lstrip("\n").split("\n")[0]
            score = max(dataset_cls.metric(score_prediction, answer) for answer in answers)
            result = {
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
                "backend": f"hf-transformers-{transformers.__version__}",
                "rope_buffer_dtype": args.rope_buffer_dtype,
            }
            append_jsonl_fsync(result_path, result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

        completed = read_resumable_jsonl(result_path)
        completed = [
            row
            for row in completed
            if args.start_index <= row["index"] < args.limit
        ]
        mean_score = sum(row["score"] for row in completed) / len(completed)
        summary = {
            "dataset": dataset_name,
            "completed": len(completed),
            "mean_score": mean_score,
            "paper_target": PAPER_TARGETS[dataset_name],
            "delta_from_paper": mean_score - PAPER_TARGETS[dataset_name],
            "rope_buffer_dtype": args.rope_buffer_dtype,
        }
        write_text_atomic(
            output_dir / f"{dataset_name}_summary.json",
            json.dumps(summary, indent=2) + "\n",
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
