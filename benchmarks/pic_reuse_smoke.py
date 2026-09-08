"""Verify that two requests with the same context reuse one stored PIC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import torch
from transformers import AutoTokenizer

from benchmarks.evaluate_hf_repro import DEFAULT_DATA, DEFAULT_TOKENIZER
from comb import COMB
from comb.supported_models import COMB_MODEL_MAPPING
from data import TEST_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    tokenizer_path = Path(args.tokenizer).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    os.environ["COMB_TOKENIZER_PATH"] = str(tokenizer_path)
    os.environ["IPC_PATH"] = f"/tmp/comb_pic_reuse_{os.getpid()}_{uuid.uuid4().hex}"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    with (Path(args.data_dir) / "hotpotqa.jsonl").open() as handle:
        row = json.loads(next(handle))
    dataset_cls = TEST_DATASETS["hotpotqa"]
    question = dataset_cls.instruction + row["input"]
    question_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}]
    )
    alternate_question_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Please answer carefully.\n" + question}]
    )
    chunk_ids = tokenizer(row["context"])["input_ids"]

    COMB_MODEL_MAPPING[str(tokenizer_path)] = args.model
    comb = COMB(
        str(tokenizer_path),
        pic_memory_utilization=0.15,
        pbc_memory_utilization=0.60,
        disable_log_stats=True,
    )
    comb.set_sampling_params(temperature=0.0, max_tokens=8)

    process_calls = 0
    processor = comb.pic_manager.chunk_processor[0]
    original_process = processor.process

    def counted_process(tokens):
        nonlocal process_calls
        process_calls += 1
        return original_process(tokens)

    processor.process = counted_process
    runs = []
    split = len(chunk_ids) // 2
    reordered_chunk_ids = chunk_ids[split:] + chunk_ids[:split]
    requests = (
        ("initial", question_ids, chunk_ids),
        ("exact_repeat", question_ids, chunk_ids),
        ("different_query_prefix", alternate_question_ids, chunk_ids),
        ("reordered_context", question_ids, reordered_chunk_ids),
    )
    for request_name, request_ids, request_chunk_ids in requests:
        prompt = {
            "input_ids": list(request_ids),
            "chunk_ids": list(request_chunk_ids),
        }
        started = time.perf_counter()
        output = comb.generate(prompt, need_store=True, use_tqdm=False)[0]
        torch.cuda.synchronize()
        runs.append(
            {
                "request": request_name,
                "question_tokens": len(request_ids),
                "token_ids": list(output.token_ids),
                "seconds": time.perf_counter() - started,
                "cache_entries": len(comb.pic_manager.cached_hash_to_picinfo),
                "chunk_process_calls": process_calls,
            }
        )

    result = {
        "model": str(Path(args.model).resolve()),
        "context_tokens": len(chunk_ids),
        "runs": runs,
        "exact_repeat_same_output": runs[0]["token_ids"] == runs[1]["token_ids"],
        "pic_reused_across_query_prefixes": runs[2]["chunk_process_calls"] == 1,
        "single_cache_entry_for_shared_context": runs[2]["cache_entries"] == 1,
        "reordered_context_encoded_separately": (
            runs[3]["chunk_process_calls"] == 2 and runs[3]["cache_entries"] == 2
        ),
    }
    if not all(
        result[key]
        for key in (
            "exact_repeat_same_output",
            "pic_reused_across_query_prefixes",
            "single_cache_entry_for_shared_context",
            "reordered_context_encoded_separately",
        )
    ):
        raise RuntimeError(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
