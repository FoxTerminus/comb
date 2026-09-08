"""Resumable LongBench accuracy evaluation using the paper's COMB/vLLM path."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.evaluate_hf_repro import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    PAPER_TARGETS,
    existing_indices,
    load_rows,
)
from comb import COMB
from comb.supported_models import COMB_MODEL_MAPPING
from data import TEST_DATASETS
from data.metrics import qa_f1_score
from training.artifact_io import (
    append_jsonl_fsync,
    read_resumable_jsonl,
    write_text_atomic,
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
    parser.add_argument("--pic-memory-utilization", type=float, default=0.15)
    parser.add_argument("--pbc-memory-utilization", type=float, default=0.60)
    parser.add_argument(
        "--ipc-path",
        help="Base path for COMB CUDA IPC. Defaults to a unique per-process path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = Path(args.tokenizer).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Local tokenizer directory not found: {tokenizer_path}")
    os.environ["COMB_TOKENIZER_PATH"] = str(tokenizer_path)
    ipc_path = args.ipc_path or f"/tmp/comb_ipc_{os.getpid()}_{uuid.uuid4().hex}"
    os.environ["IPC_PATH"] = ipc_path
    print(json.dumps({"ipc_path": ipc_path}), flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    COMB_MODEL_MAPPING[args.tokenizer] = args.model
    comb = COMB(
        args.tokenizer,
        pic_memory_utilization=args.pic_memory_utilization,
        pbc_memory_utilization=args.pbc_memory_utilization,
        disable_log_stats=True,
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
        comb.set_sampling_params(temperature=0.0, max_tokens=max_new_tokens)
        for index, row in enumerate(rows[args.start_index:], start=args.start_index):
            if index in finished:
                continue
            answers = row["answers"]
            if hasattr(dataset_cls, "correct_answers"):
                answers = dataset_cls.correct_answers.get(row["input"], answers)
            prompt = {
                "input_ids": tokenizer.apply_chat_template(
                    [{"role": "user", "content": dataset_cls.instruction + row["input"]}]
                ),
                "chunk_ids": tokenizer(row["context"])["input_ids"],
            }
            started = time.perf_counter()
            output = comb.generate(prompt, use_tqdm=False)[0]
            prediction = tokenizer.decode(
                output.token_ids, skip_special_tokens=True
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
                "context_tokens": len(prompt["chunk_ids"]),
                "generated_tokens": len(output.token_ids),
                "seconds": time.perf_counter() - started,
                "backend": "comb-vllm-0.12.0",
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
            "backend": "comb-vllm-0.12.0",
        }
        write_text_atomic(
            output_dir / f"{dataset_name}_summary.json",
            json.dumps(summary, indent=2) + "\n",
        )
        print(json.dumps(summary), flush=True)



if __name__ == "__main__":
    main()
