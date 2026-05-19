#!/usr/bin/env python
"""Run the synthetic Phonebook retrieval benchmark."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_NUM_PAIRS, DEFAULT_NUM_SAMPLES, DEFAULT_OUTPUT_DIR, MODEL_SPECS
from data import get_shard, iter_examples
from io_utils import append_jsonl, read_jsonl
from models import load_adapter
from score import score_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="phonebook_32k")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_NUM_PAIRS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-mode", choices=["uniform", "early", "middle", "late"], default="uniform")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    elif out_path.exists():
        for row in read_jsonl(out_path):
            done.add(int(row["global_index"]))

    examples = get_shard(
        iter_examples(args.num_samples, args.num_pairs, args.seed, args.position_mode),
        args.shard_id,
        args.num_shards,
    )
    print(
        f"Running {args.model} shard {args.shard_id}/{args.num_shards}: "
        f"{len(examples)} examples, {len(done)} already done",
        flush=True,
    )
    adapter = load_adapter(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    for idx, ex in enumerate(examples, start=1):
        if ex.global_index in done:
            continue
        try:
            result = adapter.generate(ex)
            scores = score_prediction(result["prediction"], ex.answer)
            record = {
                "model": args.model,
                "trained_on_prolong_64k": MODEL_SPECS[args.model].trained_on_prolong_64k,
                "global_index": ex.global_index,
                "sample_id": ex.sample_id,
                "target_position": ex.target_position,
                "target_position_frac": ex.target_position / max(ex.num_pairs - 1, 1),
                "num_pairs": ex.num_pairs,
                "target_name": ex.target_name,
                "answer": ex.answer,
                "question": ex.question,
                **result,
                **scores,
            }
        except Exception as exc:
            if not args.continue_on_error:
                raise
            record = {
                "model": args.model,
                "trained_on_prolong_64k": MODEL_SPECS[args.model].trained_on_prolong_64k,
                "global_index": ex.global_index,
                "sample_id": ex.sample_id,
                "target_position": ex.target_position,
                "target_position_frac": ex.target_position / max(ex.num_pairs - 1, 1),
                "num_pairs": ex.num_pairs,
                "target_name": ex.target_name,
                "answer": ex.answer,
                "question": ex.question,
                "prediction": "",
                "prediction_digits": "",
                "answer_digits": ex.answer,
                "exact": 0.0,
                "contains": 0.0,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "input_tokens": 0,
                "chunk_tokens": 0,
                "output_tokens": 0,
                "latency_sec": 0.0,
            }
        append_jsonl(out_path, [record])
        print(
            f"[{idx}/{len(examples)}] sample={ex.global_index} "
            f"pos={ex.target_position}/{ex.num_pairs} exact={record['exact']} "
            f"pred={record.get('prediction')!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()

