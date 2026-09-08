#!/usr/bin/env python3
"""Validate an HF Comb checkpoint without materializing its tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from training.artifact_io import write_text_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-parameters", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    model, loading_info = CombLlamaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map="meta",
        dtype="auto",
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {
        key: value
        for key, value in loading_info.items()
        if value
    }
    if problems:
        raise RuntimeError({"invalid_hf_checkpoint": problems})
    non_meta = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type != "meta"
    ]
    if non_meta:
        raise RuntimeError({"unexpected_materialized_parameters": non_meta[:20]})

    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("No safetensors weight files found")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        args.expected_parameters is not None
        and parameter_count != args.expected_parameters
    ):
        raise ValueError(
            f"Expected {args.expected_parameters} parameters, found {parameter_count}"
        )
    result = {
        "checkpoint": str(checkpoint),
        "model_class": type(model).__name__,
        "parameter_count": parameter_count,
        "safetensors_shards": len(shards),
        "total_weight_bytes": sum(path.stat().st_size for path in shards),
        "strict_loading_valid": True,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        write_text_atomic(args.output, rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
