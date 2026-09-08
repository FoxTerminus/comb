"""Measure whether Comb assigns lower target NLL to the correct context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk

from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from comb.integration.hf.CombLlamaFrozen32 import (
    CombLlamaFrozen32ForConditionalGeneration,
)
from training.artifact_io import write_text_atomic


DEFAULT_MODEL = "/data3/junhaohu/model/official_CombLlama-11B-Instruct"
DEFAULT_DATA = (
    "/data3/junhaohu/.cache/huggingface/datasets/"
    "SQuAD_meta-llama_Llama-3.1-8B-Instruct"
)
TEACHER_COLUMN = "meta-llama/Llama-3.1-8B-Instruct"
PAD_TOKEN_ID = 128004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--optimizer-step", type=int)
    parser.add_argument("--context-tokens-gt", type=int)
    parser.add_argument("--context-tokens-le", type=int)
    parser.add_argument("--target-column", default=TEACHER_COLUMN)
    parser.add_argument("--output")
    parser.add_argument(
        "--architecture", choices=("comb", "frozen32"), default="comb"
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def pad(sequence: list[int], length: int, value: int) -> list[int]:
    return sequence[:length] + [value] * max(length - len(sequence), 0)


def make_batch(
    rows: list[dict],
    context_rows: list[dict] | None = None,
    target_column: str = TEACHER_COLUMN,
) -> dict[str, torch.Tensor]:
    if context_rows is None:
        context_rows = rows
    if len(rows) != len(context_rows):
        raise ValueError("query and context row counts must match")
    chunk_length = max(len(row["chunk_ids"]) for row in context_rows)
    input_ids, shift_labels, chunk_ids, masks = [], [], [], []
    for row, context_row in zip(rows, context_rows, strict=True):
        query = row["input_ids"]
        target = row[target_column]
        combined = (query + target)[:512]
        shifted = ([-100] * (len(query) - 1) + target)[:512]
        input_ids.append(pad(combined, 512, PAD_TOKEN_ID))
        shift_labels.append(pad(shifted, 512, -100))
        chunk_ids.append(
            pad(context_row["chunk_ids"], chunk_length, PAD_TOKEN_ID)
        )
        masks.append(
            pad(context_row["cross_attention_mask"], chunk_length, 0)
        )
    return {
        "input_ids": torch.tensor(np.asarray(input_ids), device="cuda"),
        "shift_labels": torch.tensor(np.asarray(shift_labels), device="cuda"),
        "chunk_ids": torch.tensor(np.asarray(chunk_ids), device="cuda"),
        "cross_attention_mask": torch.tensor(np.asarray(masks), device="cuda"),
    }


def per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[list[float], list[int]]:
    token_losses = F.cross_entropy(
        logits.transpose(1, 2), labels, reduction="none", ignore_index=-100
    )
    valid = labels != -100
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise RuntimeError("Encountered an evaluation example without target tokens")
    losses = (token_losses * valid).sum(dim=1) / counts
    return losses.detach().cpu().tolist(), counts.detach().cpu().tolist()


def singleton_shuffled_context_rows(
    rows: list[dict], start: int, stop: int
) -> list[dict] | None:
    """Provide a real cyclic mismatch when memory limits force batch size one."""
    if stop - start != 1:
        return None
    return [rows[(start - 1) % len(rows)]]


def evaluate_model(
    model,
    dataset,
    limit: int,
    batch_size: int,
    target_column: str = TEACHER_COLUMN,
) -> dict:
    target_count = min(limit, len(dataset))
    # Adjacent SQuAD questions commonly share one paragraph.  Sample across
    # the full dataset and retain distinct tokenized contexts so a small eval
    # limit still measures context dependence instead of failing on duplicates.
    candidate_indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(len(dataset), max(target_count * 8, target_count)),
        dtype=np.int64,
    ).tolist()
    candidate_indices.extend(range(len(dataset)))
    seen_contexts = set()
    rows = []
    for index in candidate_indices:
        row = dataset[int(index)]
        signature = hashlib.sha256(
            np.asarray(row["chunk_ids"], dtype=np.int32).tobytes()
        ).digest()
        if signature in seen_contexts:
            continue
        seen_contexts.add(signature)
        rows.append(row)
        if len(rows) == target_count:
            break
    if len(rows) < 2:
        raise ValueError("context-dependency evaluation requires at least two rows")
    # Start half a dataset away to avoid adjacent SQuAD questions that commonly
    # share the same paragraph, then scan until the tokenized context differs.
    distinct_context_indices = []
    for index, row in enumerate(rows):
        for delta in range(len(rows) // 2, len(rows) // 2 + len(rows)):
            candidate = (index + delta) % len(rows)
            if candidate != index and rows[candidate]["chunk_ids"] != row["chunk_ids"]:
                distinct_context_indices.append(candidate)
                break
        else:
            raise RuntimeError("evaluation sample contains only one distinct context")

    correct_nll = shuffled_nll = 0.0
    distinct_nll = no_context_nll = 0.0
    token_count = 0
    paired_batches = []
    paired_examples = []
    distinct_batches = []
    distinct_examples = []
    no_context_batches = []
    no_context_examples = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            batch_rows = rows[start:stop]
            batch = make_batch(batch_rows, target_column=target_column)
            valid_tokens = int((batch["shift_labels"] != -100).sum())
            correct_output = model(**batch, use_cache=False)
            correct = correct_output.loss
            correct_examples, example_tokens = per_example_nll(
                correct_output.logits, batch["shift_labels"]
            )
            del correct_output
            singleton_context_rows = singleton_shuffled_context_rows(
                rows, start, stop
            )
            if singleton_context_rows is None:
                wrong_batch = dict(batch)
                wrong_batch["chunk_ids"] = batch["chunk_ids"].roll(1, dims=0)
                wrong_batch["cross_attention_mask"] = batch[
                    "cross_attention_mask"
                ].roll(1, dims=0)
            else:
                wrong_batch = make_batch(
                    batch_rows,
                    singleton_context_rows,
                    target_column=target_column,
                )
            shuffled_output = model(**wrong_batch, use_cache=False)
            shuffled = shuffled_output.loss
            shuffled_examples, wrong_example_tokens = per_example_nll(
                shuffled_output.logits, batch["shift_labels"]
            )
            del shuffled_output
            if example_tokens != wrong_example_tokens:
                raise RuntimeError("Correct and shuffled target-token counts differ")
            distinct_batch = make_batch(
                batch_rows,
                [rows[distinct_context_indices[index]] for index in range(start, stop)],
                target_column=target_column,
            )
            distinct_output = model(**distinct_batch, use_cache=False)
            distinct = distinct_output.loss
            distinct_example_nll, distinct_example_tokens = per_example_nll(
                distinct_output.logits, distinct_batch["shift_labels"]
            )
            del distinct_output
            if example_tokens != distinct_example_tokens:
                raise RuntimeError("Correct and distinct target-token counts differ")
            no_context_output = model(
                input_ids=batch["input_ids"],
                shift_labels=batch["shift_labels"],
                use_cache=False,
            )
            no_context = no_context_output.loss
            no_context_example_nll, no_context_example_tokens = per_example_nll(
                no_context_output.logits, batch["shift_labels"]
            )
            del no_context_output
            if example_tokens != no_context_example_tokens:
                raise RuntimeError("Correct and no-context target-token counts differ")
            correct_nll += float(correct) * valid_tokens
            shuffled_nll += float(shuffled) * valid_tokens
            distinct_nll += float(distinct) * valid_tokens
            no_context_nll += float(no_context) * valid_tokens
            token_count += valid_tokens
            paired_batches.append(
                {
                    "start_example": start,
                    "examples": int(batch["input_ids"].shape[0]),
                    "target_tokens": valid_tokens,
                    "correct_context_nll": float(correct),
                    "shuffled_context_nll": float(shuffled),
                    "context_nll_gap": float(shuffled) - float(correct),
                }
            )
            for offset, (correct_example, shuffled_example, tokens) in enumerate(
                zip(correct_examples, shuffled_examples, example_tokens, strict=True)
            ):
                paired_examples.append(
                    {
                        "example_index": start + offset,
                        "target_tokens": int(tokens),
                        "correct_context_nll": float(correct_example),
                        "shuffled_context_nll": float(shuffled_example),
                        "context_nll_gap": float(shuffled_example - correct_example),
                    }
                )
            distinct_batches.append(
                {
                    "start_example": start,
                    "examples": int(batch["input_ids"].shape[0]),
                    "target_tokens": valid_tokens,
                    "correct_context_nll": float(correct),
                    "distinct_context_nll": float(distinct),
                    "distinct_context_nll_gap": float(distinct) - float(correct),
                }
            )
            for offset, (correct_example, wrong_example, tokens) in enumerate(
                zip(
                    correct_examples,
                    distinct_example_nll,
                    example_tokens,
                    strict=True,
                )
            ):
                example_index = start + offset
                distinct_examples.append(
                    {
                        "example_index": example_index,
                        "wrong_context_example_index": distinct_context_indices[
                            example_index
                        ],
                        "target_tokens": int(tokens),
                        "correct_context_nll": float(correct_example),
                        "distinct_context_nll": float(wrong_example),
                        "distinct_context_nll_gap": float(
                            wrong_example - correct_example
                        ),
                    }
                )
            no_context_batches.append(
                {
                    "start_example": start,
                    "examples": int(batch["input_ids"].shape[0]),
                    "target_tokens": valid_tokens,
                    "correct_context_nll": float(correct),
                    "no_context_nll": float(no_context),
                    "no_context_nll_gap": float(no_context) - float(correct),
                }
            )
            for offset, (correct_example, absent_example, tokens) in enumerate(
                zip(
                    correct_examples,
                    no_context_example_nll,
                    example_tokens,
                    strict=True,
                )
            ):
                no_context_examples.append(
                    {
                        "example_index": start + offset,
                        "target_tokens": int(tokens),
                        "correct_context_nll": float(correct_example),
                        "no_context_nll": float(absent_example),
                        "no_context_nll_gap": float(
                            absent_example - correct_example
                        ),
                    }
                )

    correct_nll /= token_count
    shuffled_nll /= token_count
    distinct_nll /= token_count
    no_context_nll /= token_count
    gates = {
        name: float(parameter.detach().float().tanh().abs().mean())
        for name, parameter in model.named_parameters()
        if name.endswith("cross_attn_attn_gate")
        or name.endswith("cross_attn_mlp_gate")
        or name.endswith("context_gate")
    }
    batch_gaps = np.asarray(
        [batch["context_nll_gap"] for batch in paired_batches], dtype=np.float64
    )
    batch_gap_standard_error = (
        float(batch_gaps.std(ddof=1) / math.sqrt(len(batch_gaps)))
        if len(batch_gaps) > 1
        else None
    )
    example_gaps = np.asarray(
        [example["context_nll_gap"] for example in paired_examples], dtype=np.float64
    )
    example_gap_standard_error = (
        float(example_gaps.std(ddof=1) / math.sqrt(len(example_gaps)))
        if len(example_gaps) > 1
        else None
    )
    distinct_batch_gaps = np.asarray(
        [batch["distinct_context_nll_gap"] for batch in distinct_batches],
        dtype=np.float64,
    )
    distinct_example_gaps = np.asarray(
        [example["distinct_context_nll_gap"] for example in distinct_examples],
        dtype=np.float64,
    )
    no_context_example_gaps = np.asarray(
        [example["no_context_nll_gap"] for example in no_context_examples],
        dtype=np.float64,
    )
    return {
        "examples": len(rows),
        "target_tokens": token_count,
        "correct_context_nll": correct_nll,
        "shuffled_context_nll": shuffled_nll,
        "context_nll_gap": shuffled_nll - correct_nll,
        "paired_batch_gap_mean": float(batch_gaps.mean()),
        "paired_batch_count": len(paired_batches),
        "paired_batch_gap_standard_error": batch_gap_standard_error,
        "positive_batch_fraction": float((batch_gaps > 0).mean()),
        "paired_batches": paired_batches,
        "paired_example_gap_mean": float(example_gaps.mean()),
        "paired_example_gap_standard_error": example_gap_standard_error,
        "positive_example_fraction": float((example_gaps > 0).mean()),
        "paired_examples": paired_examples,
        "distinct_context_nll": distinct_nll,
        "distinct_context_nll_gap": distinct_nll - correct_nll,
        "distinct_batch_gap_mean": float(distinct_batch_gaps.mean()),
        "distinct_batch_gap_standard_error": (
            float(distinct_batch_gaps.std(ddof=1) / math.sqrt(len(distinct_batch_gaps)))
            if len(distinct_batch_gaps) > 1
            else None
        ),
        "distinct_positive_batch_fraction": float(
            (distinct_batch_gaps > 0).mean()
        ),
        "distinct_batches": distinct_batches,
        "distinct_example_gap_mean": float(distinct_example_gaps.mean()),
        "distinct_example_gap_standard_error": (
            float(
                distinct_example_gaps.std(ddof=1)
                / math.sqrt(len(distinct_example_gaps))
            )
            if len(distinct_example_gaps) > 1
            else None
        ),
        "distinct_positive_example_fraction": float(
            (distinct_example_gaps > 0).mean()
        ),
        "distinct_examples": distinct_examples,
        "no_context_nll": no_context_nll,
        "no_context_nll_gap": no_context_nll - correct_nll,
        "no_context_positive_example_fraction": float(
            (no_context_example_gaps > 0).mean()
        ),
        "no_context_batches": no_context_batches,
        "no_context_examples": no_context_examples,
        "mean_abs_tanh_gate": sum(gates.values()) / len(gates),
        "gates": gates,
    }


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.data)
    source_fingerprint = str(getattr(dataset, "_fingerprint", "unavailable"))
    source_rows = len(dataset)
    if args.context_tokens_gt is not None or args.context_tokens_le is not None:
        if "token_count" not in dataset.column_names:
            raise ValueError("context-token filtering requires token_count")
        token_counts = (
            dataset.data.table.column("token_count")
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        selected = np.ones(len(token_counts), dtype=bool)
        if args.context_tokens_gt is not None:
            selected &= token_counts > args.context_tokens_gt
        if args.context_tokens_le is not None:
            selected &= token_counts <= args.context_tokens_le
        source_indices = np.flatnonzero(selected)[: args.limit]
        if len(source_indices) < args.limit:
            raise ValueError(
                f"only {len(source_indices)} rows match the context-token filter"
            )
        dataset = dataset.select(source_indices.tolist())
    else:
        source_indices = np.arange(min(args.limit, len(dataset)))
    selected_lengths = [int(dataset[index]["token_count"]) for index in range(min(args.limit, len(dataset)))]
    model_class = (
        CombLlamaFrozen32ForConditionalGeneration
        if args.architecture == "frozen32"
        else CombLlamaForConditionalGeneration
    )
    model = model_class.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).cuda().eval()
    result = {
        "model": args.model,
        "data": args.data,
        "source_dataset_fingerprint": source_fingerprint,
        "source_dataset_rows": source_rows,
        "selected_source_indices": source_indices.tolist(),
        "context_token_filter": {
            "gt": args.context_tokens_gt,
            "le": args.context_tokens_le,
        },
        "selected_context_tokens": {
            "minimum": min(selected_lengths),
            "maximum": max(selected_lengths),
            "mean": sum(selected_lengths) / len(selected_lengths),
        },
        "target_column": args.target_column,
        **evaluate_model(
            model,
            dataset,
            args.limit,
            args.batch_size,
            target_column=args.target_column,
        ),
    }
    if args.optimizer_step is not None:
        result["optimizer_step"] = args.optimizer_step
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        write_text_atomic(Path(args.output), rendered + "\n")


if __name__ == "__main__":
    main()
