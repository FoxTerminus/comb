#!/usr/bin/env python3
"""Compare two fixed held-out validation-NLL artifacts without imposing a trend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.artifact_io import write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = json.loads(args.current.read_text())
    previous = json.loads(args.previous.read_text())
    current_manifest = current["validation_manifest"]
    previous_manifest = previous["validation_manifest"]
    identity_keys = (
        "source_fingerprint",
        "selected_id_sha256",
        "saved_fingerprint",
        "examples",
    )
    if any(current_manifest[key] != previous_manifest[key] for key in identity_keys):
        raise RuntimeError("validation sample identity changed between milestones")
    teacher_keys = ("teacher_column", "teacher_token_sha256", "saved_fingerprint")
    current_teacher = current["teacher_manifest"]
    previous_teacher = previous["teacher_manifest"]
    if any(current_teacher[key] != previous_teacher[key] for key in teacher_keys):
        raise RuntimeError("validation teacher targets changed between milestones")
    if int(current["target_tokens"]) != int(previous["target_tokens"]):
        raise RuntimeError("validation target-token count changed between milestones")
    current_batches = current["batches"]
    previous_batches = previous["batches"]
    if len(current_batches) != len(previous_batches) or any(
        (item[0]["start_example"], item[0]["examples"], item[0]["target_tokens"])
        != (item[1]["start_example"], item[1]["examples"], item[1]["target_tokens"])
        for item in zip(current_batches, previous_batches, strict=True)
    ):
        raise RuntimeError("validation batch layout changed between milestones")
    batch_deltas = [
        float(current_batch["nll"]) - float(previous_batch["nll"])
        for current_batch, previous_batch in zip(
            current_batches, previous_batches, strict=True
        )
    ]
    batch_delta_mean = sum(batch_deltas) / len(batch_deltas)
    batch_delta_standard_error = (
        math.sqrt(
            sum((value - batch_delta_mean) ** 2 for value in batch_deltas)
            / (len(batch_deltas) - 1)
        )
        / math.sqrt(len(batch_deltas))
        if len(batch_deltas) > 1
        else None
    )
    report = {
        "scope": "descriptive held-out trend; no pass/fail threshold",
        "steps": [previous.get("optimizer_step"), current.get("optimizer_step")],
        "examples": int(current["examples"]),
        "target_tokens": int(current["target_tokens"]),
        "previous_validation_nll": float(previous["validation_nll"]),
        "current_validation_nll": float(current["validation_nll"]),
        "validation_nll_delta": float(current["validation_nll"])
        - float(previous["validation_nll"]),
        "paired_batch_nll_delta_mean": batch_delta_mean,
        "paired_batch_nll_delta_standard_error": batch_delta_standard_error,
        "paired_batch_improved_fraction": sum(
            value < 0 for value in batch_deltas
        )
        / len(batch_deltas),
        "previous_validation_perplexity": float(previous["validation_perplexity"]),
        "current_validation_perplexity": float(current["validation_perplexity"]),
    }
    if args.reference:
        reference = json.loads(args.reference.read_text())
        reference_manifest = reference["validation_manifest"]
        reference_teacher = reference["teacher_manifest"]
        if any(
            current_manifest[key] != reference_manifest[key]
            for key in identity_keys
        ) or any(
            current_teacher[key] != reference_teacher[key]
            for key in teacher_keys
        ) or int(current["target_tokens"]) != int(reference["target_tokens"]):
            raise RuntimeError("validation reference sample identity changed")
        report["reference_validation_nll"] = float(reference["validation_nll"])
        report["current_minus_reference_nll"] = float(
            current["validation_nll"]
        ) - float(reference["validation_nll"])
    write_text_atomic(args.output, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
