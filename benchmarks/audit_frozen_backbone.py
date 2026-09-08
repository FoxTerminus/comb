#!/usr/bin/env python3
"""Verify that a trained Comb HF checkpoint preserved every frozen Llama weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.artifact_io import write_text_atomic

from safetensors import safe_open
import torch


EXPECTED_BASE_PARAMETERS = 8_030_261_248
EXPECTED_COMB_PARAMETERS = 12_045_391_888


class ShardedSafetensors:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def comb_name(base_name: str) -> str:
    if base_name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if base_name.startswith("model."):
        return "language_model." + base_name
    raise ValueError(f"unexpected base parameter name: {base_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comb", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comb = ShardedSafetensors(args.comb.resolve())
    base = ShardedSafetensors(args.base.resolve())
    base_parameters = 0
    mismatched = []
    missing = []
    for base_name in sorted(base.weight_map):
        expected_name = comb_name(base_name)
        if expected_name not in comb.weight_map:
            missing.append(expected_name)
            continue
        base_tensor = base.tensor(base_name)
        comb_tensor = comb.tensor(expected_name)
        base_parameters += base_tensor.numel()
        if not torch.equal(base_tensor, comb_tensor):
            mismatched.append(expected_name)

    base_embedding = base.tensor("model.embed_tokens.weight")
    chunk_embedding = comb.tensor("chunk_model.embed_tokens.weight")
    chunk_embedding_equal = torch.equal(base_embedding, chunk_embedding)
    comb_parameters = sum(
        comb.tensor(name).numel() for name in sorted(comb.weight_map)
    )
    result = {
        "comb_checkpoint": str(args.comb.resolve()),
        "base_checkpoint": str(args.base.resolve()),
        "base_tensors_compared": len(base.weight_map),
        "base_elements_compared": base_parameters,
        "missing_tensors": missing,
        "mismatched_tensors": mismatched,
        "decoder_backbone_exact_equal": not missing and not mismatched,
        "chunk_embedding_exact_equal_to_base_embedding": chunk_embedding_equal,
        "checkpoint_tensors": len(comb.weight_map),
        "checkpoint_elements": comb_parameters,
    }
    if base_parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(f"unexpected base parameter count: {base_parameters}")
    if comb_parameters != EXPECTED_COMB_PARAMETERS:
        raise RuntimeError(f"unexpected Comb parameter count: {comb_parameters}")
    if not result["decoder_backbone_exact_equal"] or not chunk_embedding_equal:
        raise RuntimeError(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
