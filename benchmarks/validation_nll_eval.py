#!/usr/bin/env python3
"""Compute held-out, token-weighted NLL without changing the training loop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from training.artifact_io import write_text_atomic


DEFAULT_DATA = Path(
    "/data3/junhaohu/checkpoints/Comb_validation/"
    "squad_v2_validation_llama_teacher_vllm0101_tp2_seed42_256"
)
PAD_TOKEN_ID = 128004
INPUT_LENGTH = 512
TEACHER_COLUMN = "meta-llama/Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-column", default=TEACHER_COLUMN)
    parser.add_argument("--optimizer-step", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def pad(values: list[int], length: int, value: int) -> list[int]:
    return values[:length] + [value] * max(length - len(values), 0)


def make_batch(
    rows: list[dict],
    device: torch.device | str,
    target_column: str = "labels",
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("validation batch must not be empty")
    chunk_length = max(len(row["chunk_ids"]) for row in rows)
    input_ids, shift_labels, chunk_ids, masks = [], [], [], []
    for row in rows:
        query = row["input_ids"]
        target = row[target_column]
        combined = (query + target)[:INPUT_LENGTH]
        shifted = ([-100] * (len(query) - 1) + target)[:INPUT_LENGTH]
        input_ids.append(pad(combined, INPUT_LENGTH, PAD_TOKEN_ID))
        shift_labels.append(pad(shifted, INPUT_LENGTH, -100))
        chunk_ids.append(pad(row["chunk_ids"], chunk_length, PAD_TOKEN_ID))
        masks.append(pad(row["cross_attention_mask"], chunk_length, 0))
    result = {
        "input_ids": torch.tensor(np.asarray(input_ids), device=device),
        "shift_labels": torch.tensor(np.asarray(shift_labels), device=device),
        "chunk_ids": torch.tensor(np.asarray(chunk_ids), device=device),
        "cross_attention_mask": torch.tensor(np.asarray(masks), device=device),
    }
    if not torch.all((result["shift_labels"] != -100).any(dim=1)):
        raise RuntimeError("validation example has no target tokens after truncation")
    return result


def evaluate_model(
    model, dataset, batch_size: int, target_column: str
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    total_nll = 0.0
    total_tokens = 0
    batches = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            rows = [dataset[index] for index in range(start, stop)]
            batch = make_batch(rows, device, target_column)
            output = model(**batch, use_cache=False)
            loss = float(output.loss)
            tokens = int((batch["shift_labels"] != -100).sum())
            if not math.isfinite(loss) or tokens < 1:
                raise RuntimeError("non-finite validation loss or empty target batch")
            total_nll += loss * tokens
            total_tokens += tokens
            batches.append(
                {
                    "start_example": start,
                    "examples": stop - start,
                    "target_tokens": tokens,
                    "nll": loss,
                }
            )
            del output, batch
    validation_nll = total_nll / total_tokens
    return {
        "examples": len(dataset),
        "target_tokens": total_tokens,
        "validation_nll": validation_nll,
        "validation_perplexity": math.exp(validation_nll),
        "batches": batches,
    }


def main() -> None:
    args = parse_args()
    teacher_manifest_path = args.data / "comb_validation_teacher_manifest.json"
    if teacher_manifest_path.is_file():
        teacher_manifest = json.loads(teacher_manifest_path.read_text())
        manifest = teacher_manifest["source_manifest"]
    else:
        teacher_manifest = None
        manifest = json.loads(
            (args.data / "comb_validation_manifest.json").read_text()
        )
    dataset = load_from_disk(str(args.data))
    if len(dataset) != int(manifest["examples"]):
        raise RuntimeError("validation dataset and manifest row counts differ")
    if args.target_column not in dataset.column_names:
        raise RuntimeError(f"validation target column is missing: {args.target_column}")
    model = CombLlamaForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()
    result = {
        "scope": "held-out sidecar metric; excluded from training and paper-score thresholds",
        "model": args.model,
        "data": str(args.data),
        "prepared_dataset_fingerprint": str(dataset._fingerprint),
        "validation_manifest": manifest,
        "teacher_manifest": teacher_manifest,
        "target_column": args.target_column,
        **evaluate_model(model, dataset, args.batch_size, args.target_column),
    }
    if args.optimizer_step is not None:
        result["optimizer_step"] = args.optimizer_step
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        write_text_atomic(args.output, rendered + "\n")


if __name__ == "__main__":
    main()
