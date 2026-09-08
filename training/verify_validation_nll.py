#!/usr/bin/env python3
"""Verify provenance and numerical integrity of a sidecar validation-NLL run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_SOURCE_FINGERPRINT = "98fb35dfc392a2a7"
EXPECTED_SAVED_FINGERPRINT = "14afa754b0ca1a0b"
EXPECTED_ID_SHA256 = "931ba3db78f3d5a077acac326fd680b53e58b70ecd2d13a1fd73492fb050eb75"
EXPECTED_TEACHER_SHA256 = "4c96f2ae6da29f0bcced73f7cf8c5f1a229419b776359a01c6864b25f917e7c6"
EXPECTED_TARGET_TOKENS = 24_182
EXPECTED_TARGET_COLUMN = "meta-llama/Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-examples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.path.read_text())
    manifest = payload["validation_manifest"]
    teacher = payload["teacher_manifest"]
    checks = {
        "scope": payload["scope"].startswith("held-out sidecar metric"),
        "step": args.expected_step is None
        or int(payload["optimizer_step"]) == args.expected_step,
        "examples": int(payload["examples"]) == args.expected_examples,
        "source_rows": int(manifest["source_rows"]) == 11_873,
        "source_fingerprint": manifest["source_fingerprint"]
        == EXPECTED_SOURCE_FINGERPRINT,
        "saved_fingerprint": payload["prepared_dataset_fingerprint"]
        == EXPECTED_SAVED_FINGERPRINT
        and teacher["saved_fingerprint"] == EXPECTED_SAVED_FINGERPRINT,
        "selected_ids": manifest["selected_id_sha256"] == EXPECTED_ID_SHA256,
        "teacher_provenance": payload["target_column"] == EXPECTED_TARGET_COLUMN
        and teacher["teacher_column"] == EXPECTED_TARGET_COLUMN
        and teacher["teacher_token_sha256"] == EXPECTED_TEACHER_SHA256
        and teacher["generation"]
        == {
            "backend": "vllm",
            "vllm_version": "0.10.1.1",
            "python_version": "3.9.25",
            "torch_version": "2.7.1",
            "transformers_version": "4.55.2",
            "tensor_parallel_size": 2,
            "temperature": 0,
            "max_tokens": 512,
            "max_model_len": 2048,
        },
        "target_tokens": int(payload["target_tokens"]) == EXPECTED_TARGET_TOKENS,
        "nll": math.isfinite(float(payload["validation_nll"]))
        and float(payload["validation_nll"]) > 0,
        "perplexity": math.isclose(
            float(payload["validation_perplexity"]),
            math.exp(float(payload["validation_nll"])),
            rel_tol=1e-12,
        ),
        "batch_coverage": sum(int(item["examples"]) for item in payload["batches"])
        == args.expected_examples,
        "batch_tokens": sum(int(item["target_tokens"]) for item in payload["batches"])
        == int(payload["target_tokens"]),
        "finite_batches": all(
            math.isfinite(float(item["nll"])) and int(item["target_tokens"]) > 0
            for item in payload["batches"]
        ),
    }
    report = {"path": str(args.path), "checks": checks, "passed": all(checks.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
