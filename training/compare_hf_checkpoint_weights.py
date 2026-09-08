"""Compare every tensor in two sharded Hugging Face checkpoints exactly."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from safetensors import safe_open
import torch

from training.artifact_io import write_text_atomic


def load_index(root: Path) -> dict[str, str]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid safetensors index: {root}")
    return weight_map


def compare(left: Path, right: Path) -> dict:
    left = left.resolve()
    right = right.resolve()
    left_map = load_index(left)
    right_map = load_index(right)
    if set(left_map) != set(right_map):
        return {
            "left": str(left),
            "right": str(right),
            "passed": False,
            "missing_from_left": sorted(set(right_map) - set(left_map)),
            "missing_from_right": sorted(set(left_map) - set(right_map)),
        }
    unequal = []
    shape_or_dtype_mismatches = []
    elements = 0
    with ExitStack() as stack:
        left_files = {
            name: stack.enter_context(safe_open(left / name, framework="pt", device="cpu"))
            for name in set(left_map.values())
        }
        right_files = {
            name: stack.enter_context(safe_open(right / name, framework="pt", device="cpu"))
            for name in set(right_map.values())
        }
        for name in sorted(left_map):
            left_tensor = left_files[left_map[name]].get_tensor(name)
            right_tensor = right_files[right_map[name]].get_tensor(name)
            elements += left_tensor.numel()
            if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
                shape_or_dtype_mismatches.append(name)
            elif not torch.equal(left_tensor, right_tensor):
                unequal.append(name)
    return {
        "left": str(left),
        "right": str(right),
        "tensors": len(left_map),
        "elements": elements,
        "shape_or_dtype_mismatches": shape_or_dtype_mismatches,
        "unequal_tensors": unequal,
        "passed": not shape_or_dtype_mismatches and not unequal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.left, args.right)
    write_text_atomic(args.output, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("HF checkpoints differ")


if __name__ == "__main__":
    main()
