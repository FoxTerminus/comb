"""Export the full BF16 ``module`` stored in a released-script rank checkpoint.

The released DeepSpeed pseudo-TP checkpoint currently materializes a complete
unsharded module state in every MP-rank model-state file.  Reusing the shard
layout of a previously verified HF export lets milestone evaluation proceed
without temporarily claiming a second GPU solely for consolidation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("template_hf", type=Path)
    parser.add_argument("output_hf", type=Path)
    parser.add_argument("--expected-parameters", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    template = args.template_hf.resolve()
    output = args.output_hf.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if checkpoint.name != "mp_rank_00_model_states.pt":
        raise ValueError("Expected the rank-0 model-state checkpoint")

    index_path = template / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    state = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=False
    )["module"]
    if set(state) != set(weight_map):
        missing = sorted(set(weight_map) - set(state))
        extra = sorted(set(state) - set(weight_map))
        raise RuntimeError(
            f"HF/template key mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    parameter_count = sum(tensor.numel() for tensor in state.values())
    if parameter_count != args.expected_parameters:
        raise RuntimeError(
            f"Expected {args.expected_parameters} parameters, got {parameter_count}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        model_files = set(weight_map.values())
        for source in template.iterdir():
            if source.name in model_files or source.name == index_path.name:
                continue
            if source.is_file():
                shutil.copy2(source, temporary / source.name)

        for shard_name in sorted(model_files):
            shard = {
                name: state[name].contiguous()
                for name, filename in weight_map.items()
                if filename == shard_name
            }
            save_file(shard, temporary / shard_name, metadata={"format": "pt"})
            print(f"saved {shard_name}: {len(shard)} tensors", flush=True)
        shutil.copy2(index_path, temporary / index_path.name)

        checked = 0
        for shard_name in sorted(model_files):
            with safe_open(temporary / shard_name, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    source = state[name]
                    if (
                        tensor.shape != source.shape
                        or tensor.dtype != source.dtype
                        or not torch.equal(tensor, source)
                    ):
                        raise RuntimeError(f"Export verification failed for {name}")
                    checked += 1
        if checked != len(state):
            raise RuntimeError(f"Verified {checked} tensors, expected {len(state)}")

        os.replace(temporary, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "tensors": checked,
                    "parameters": parameter_count,
                    "source": str(checkpoint),
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
