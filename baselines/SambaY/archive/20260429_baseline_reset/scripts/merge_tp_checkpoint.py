"""Merge SambaY tensor-parallel checkpoint shards into a HF-style checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from baselines.SambaY.models.SambaY import SambaYForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Merge SambaY TP shards")
    parser.add_argument("--tp-checkpoint-dir", required=True, help="Directory containing tp_rank_*.pt files")
    parser.add_argument("--base-model-path", required=True, help="Unsharded SambaY config/checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Merged save_pretrained output directory")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def load_shards(path: Path) -> list[dict]:
    shard_paths = sorted(path.glob("tp_rank_*.pt"), key=lambda item: int(re.search(r"tp_rank_(\d+)", item.name).group(1)))
    if not shard_paths:
        raise FileNotFoundError(f"No tp_rank_*.pt shards found under {path}")
    return [torch.load(shard_path, map_location="cpu", weights_only=False)["model_state_dict"] for shard_path in shard_paths]


def merge_gated_pair_column(shards: list[torch.Tensor]) -> torch.Tensor:
    gates = []
    values = []
    for shard in shards:
        gate, value = shard.chunk(2, dim=0)
        gates.append(gate)
        values.append(value)
    return torch.cat([torch.cat(gates, dim=0), torch.cat(values, dim=0)], dim=0)


def merge_tensor(key: str, target: torch.Tensor, shard_values: list[torch.Tensor]) -> torch.Tensor:
    first = shard_values[0]
    if all(value.shape == target.shape for value in shard_values):
        return first

    tp_size = len(shard_values)
    if first.ndim == target.ndim and target.ndim >= 1:
        dim0_match = target.shape[0] == first.shape[0] * tp_size and target.shape[1:] == first.shape[1:]
        if dim0_match:
            is_mamba_gated_pair = (
                key.startswith("model.self_decoder.")
                and key.endswith("token_mixer.in_proj.weight")
                and target.ndim == 2
                and target.shape[0] == 4 * target.shape[1]
            )
            if is_mamba_gated_pair:
                return merge_gated_pair_column(shard_values)
            return torch.cat(shard_values, dim=0)

    if first.ndim == target.ndim and target.ndim >= 2:
        dim1_match = (
            target.shape[0] == first.shape[0]
            and target.shape[1] == first.shape[1] * tp_size
            and target.shape[2:] == first.shape[2:]
        )
        if dim1_match:
            return torch.cat(shard_values, dim=1)

    shapes = [tuple(value.shape) for value in shard_values]
    raise ValueError(f"Cannot merge {key}: target={tuple(target.shape)}, shards={shapes}")


def cast_state_dict(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    casted = {}
    for key, value in state_dict.items():
        casted[key] = value.to(dtype=dtype) if torch.is_floating_point(value) else value
    return casted


def merge_tp_checkpoint_to_output(
    tp_checkpoint_dir: str | Path,
    base_model_path: str,
    output_dir: str | Path,
    dtype_name: str = "bfloat16",
) -> None:
    tp_checkpoint_dir = Path(tp_checkpoint_dir)
    output_dir = Path(output_dir)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]

    print(f"Loading base SambaY model from {base_model_path}")
    model = SambaYForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    target_state = model.state_dict()
    shard_states = load_shards(tp_checkpoint_dir)

    merged = {}
    for key, target_value in target_state.items():
        if key not in shard_states[0]:
            raise KeyError(f"Key {key} missing from TP shard state dict")
        shard_values = [state[key] for state in shard_states]
        merged[key] = merge_tensor(key, target_value, shard_values)

    missing, unexpected = model.load_state_dict(cast_state_dict(merged, dtype), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Merged state_dict mismatch: missing={missing}, unexpected={unexpected}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    metadata = {
        "source_tp_checkpoint_dir": str(tp_checkpoint_dir),
        "base_model_path": base_model_path,
        "tp_size": len(shard_states),
        "dtype": dtype_name,
    }
    (output_dir / "merge_tp_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Merged SambaY checkpoint saved to {output_dir}")


def main() -> None:
    args = parse_args()
    merge_tp_checkpoint_to_output(
        tp_checkpoint_dir=args.tp_checkpoint_dir,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()
