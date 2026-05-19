#!/usr/bin/env python
"""Run a sharded full LoCoMo QA benchmark."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from config import ANS_TOKENS_PER_QUESTION, DEFAULT_OUTPUT_DIR, MODEL_SPECS
from data import get_shard, iter_examples
from io_utils import append_jsonl, read_jsonl
from models import load_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=ANS_TOKENS_PER_QUESTION)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_dir)
        / args.run_name
        / "predictions"
        / f"{args.model}.shard{args.shard_id}-of-{args.num_shards}.jsonl"
    )


def main() -> None:
    args = parse_args()
    out_path = output_path(args)
    done = set()
    if out_path.exists() and not args.overwrite:
        for row in read_jsonl(out_path):
            done.add((row["sample_id"], int(row["qa_index"])))
    elif out_path.exists() and args.overwrite:
        out_path.unlink()

    examples = get_shard(iter_examples(args.data_file), args.shard_id, args.num_shards)
    if args.limit is not None:
        examples = examples[: args.limit]

    print(
        f"Running {args.model} shard {args.shard_id}/{args.num_shards}: "
        f"{len(examples)} examples, {len(done)} already done"
    )
    adapter = load_adapter(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    for i, ex in enumerate(examples, start=1):
        key = (ex.sample_id, ex.qa_index)
        if key in done:
            continue
        try:
            result = adapter.generate(ex)
            record = {
                "model": args.model,
                "trained_on_prolong_64k": MODEL_SPECS[args.model].trained_on_prolong_64k,
                "global_index": ex.global_index,
                "sample_id": ex.sample_id,
                "qa_index": ex.qa_index,
                "category": ex.category,
                "question": ex.question,
                "answer": ex.answer,
                "evidence": ex.evidence,
                **result,
            }
        except Exception as exc:
            if not args.continue_on_error:
                raise
            record = {
                "model": args.model,
                "trained_on_prolong_64k": MODEL_SPECS[args.model].trained_on_prolong_64k,
                "global_index": ex.global_index,
                "sample_id": ex.sample_id,
                "qa_index": ex.qa_index,
                "category": ex.category,
                "question": ex.question,
                "answer": ex.answer,
                "evidence": ex.evidence,
                "prediction": "",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "input_tokens": 0,
                "chunk_tokens": 0,
                "output_tokens": 0,
                "latency_sec": 0.0,
            }
        append_jsonl(out_path, [record])
        print(
            f"[{i}/{len(examples)}] {ex.sample_id} qa={ex.qa_index} "
            f"cat={ex.category} out_tokens={record.get('output_tokens', 0)}"
        )


if __name__ == "__main__":
    main()
