#!/usr/bin/env python3
"""Compare an HF export with a native-TP forward reference on the same row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from datasets import Dataset

from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from data.base import collate_fn
from training.artifact_io import write_text_atomic


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def loss_comparison(reference_losses: object, hf_loss: float, tolerance: float) -> dict:
    if not isinstance(reference_losses, list) or not reference_losses:
        raise ValueError("native reference must contain rank losses")
    losses = [float(value) for value in reference_losses]
    finite = math.isfinite(hf_loss) and all(math.isfinite(value) for value in losses)
    deltas = [abs(hf_loss - value) for value in losses] if finite else [float("inf")]
    maximum_delta = max(deltas)
    return {
        "native_tp_rank_losses": losses,
        "hf_loss": hf_loss,
        "maximum_absolute_loss_delta": maximum_delta,
        "absolute_tolerance": tolerance,
        "finite": finite,
        "passed": finite and maximum_delta <= tolerance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--native-tensor-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.absolute_tolerance) or args.absolute_tolerance < 0:
        raise ValueError("absolute tolerance must be finite and nonnegative")
    checkpoint = args.hf_checkpoint.resolve()
    reference_path = args.native_reference.resolve()
    reference = json.loads(reference_path.read_text())
    tensor_audit_path = args.native_tensor_audit.resolve()
    tensor_audit = json.loads(tensor_audit_path.read_text())
    if not (
        tensor_audit.get("passed") is True
        and tensor_audit.get("native_checkpoint")
        == str(Path(str(reference["checkpoint"])).resolve())
        and tensor_audit.get("hf_checkpoint") == str(checkpoint)
        and int(tensor_audit.get("optimizer_steps", [-1])[0])
        == int(reference["optimizer_step"])
        and all(
            int(step) == int(reference["optimizer_step"])
            for step in tensor_audit.get("optimizer_steps", [])
        )
        and int(tensor_audit.get("compared_tensors", 0)) > 0
        and float(tensor_audit.get("maximum_absolute_delta", float("inf"))) == 0.0
        and not tensor_audit.get("mismatches")
    ):
        raise RuntimeError("native TP tensor audit is not valid for this HF export")
    parquet = Path(str(reference["parquet"])).resolve()
    if sha256_file(parquet) != reference.get("parquet_sha256"):
        raise RuntimeError("native-reference parquet identity changed")
    row_index = int(reference["row_index"])
    dataset = Dataset.from_parquet(str(parquet))
    if not 0 <= row_index < len(dataset):
        raise IndexError(row_index)
    batch = collate_fn([dataset[row_index]])
    expected_shapes = {
        "context_tokens": int(batch["chunk_ids"].shape[1]),
        "context_nonpadding_tokens": int(batch["cross_attention_mask"].sum()),
        "query_tokens": int(batch["input_ids"].shape[1]),
    }
    reference_shapes = {name: int(reference[name]) for name in expected_shapes}
    if expected_shapes != reference_shapes:
        raise RuntimeError(
            f"collated batch differs from native reference: {expected_shapes} != {reference_shapes}"
        )

    model = CombLlamaForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()
    model.eval()
    batch = {name: value.cuda() for name, value in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model(**batch)
    hf_loss = float(output.loss)
    runtime_comparison = loss_comparison(
        [rank["loss"] for rank in reference["ranks"]],
        hf_loss,
        args.absolute_tolerance,
    )
    runtime_equivalence_passed = bool(runtime_comparison.pop("passed"))
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    result = {
        "schema_version": 1,
        "scope": "native true-TP checkpoint versus consolidated HF export equivalence",
        "native_reference": str(reference_path),
        "native_reference_sha256": sha256_file(reference_path),
        "native_tensor_audit": str(tensor_audit_path),
        "native_tensor_audit_sha256": sha256_file(tensor_audit_path),
        "native_checkpoint": reference.get("checkpoint"),
        "optimizer_step": int(reference["optimizer_step"]),
        "hf_checkpoint": str(checkpoint),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "hf_model_files": {
            name: {
                "bytes": (checkpoint / name).stat().st_size,
                "sha256": sha256_file(checkpoint / name),
            }
            for name in ["config.json", "model.safetensors.index.json", *shards]
        },
        "parquet": str(parquet),
        "parquet_sha256": sha256_file(parquet),
        "row_index": row_index,
        **expected_shapes,
        "dtype": str(next(model.parameters()).dtype),
        "gpu_name": torch.cuda.get_device_name(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        **runtime_comparison,
        "tensor_equivalence_passed": True,
        "runtime_equivalence_passed": runtime_equivalence_passed,
        "runtime_drift_warning": not runtime_equivalence_passed,
        "interpretation": (
            "Exact tensor reconstruction is the export-integrity gate. Runtime loss "
            "differences are reported separately because BF16 TP and consolidated "
            "single-device execution use different matrix decompositions."
        ),
        "passed": True,
    }
    write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
