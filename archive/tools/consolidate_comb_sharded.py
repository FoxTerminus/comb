#!/usr/bin/env python3
"""Consolidate Comb FSDP/DTensor rank checkpoints into a full checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sharded-dir", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def shard_dim(dtensor: Any) -> int | None:
    placements = getattr(dtensor, "placements", None)
    if not placements:
        return None
    if len(placements) != 1:
        raise ValueError(f"Unsupported placements: {placements}")
    placement = placements[0]
    if placement.__class__.__name__ == "Shard":
        return int(placement.dim)
    return None


def local_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_local"):
        return value.to_local().detach().cpu()
    if torch.is_tensor(value):
        return value.detach().cpu()
    raise TypeError(f"Unsupported state value type: {type(value)}")


def consolidate(sharded_dir: Path, output: Path) -> None:
    rank_paths = sorted(sharded_dir.glob("rank_*.pt"))
    if not rank_paths:
        raise FileNotFoundError(f"No rank_*.pt files found in {sharded_dir}")

    metadata_path = sharded_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    expected_world = int(metadata.get("world_size", len(rank_paths)))
    if len(rank_paths) != expected_world:
        raise ValueError(f"Expected {expected_world} rank files, found {len(rank_paths)}")

    print(f"Loading {len(rank_paths)} rank checkpoints from {sharded_dir}", flush=True)
    ckpts = [torch.load(path, map_location="cpu", weights_only=False) for path in rank_paths]
    state_dicts = [ckpt["model"] for ckpt in ckpts]
    keys = list(state_dicts[0].keys())
    full_state: dict[str, torch.Tensor] = {}

    for i, key in enumerate(keys, start=1):
        values = [sd[key] for sd in state_dicts]
        dim = shard_dim(values[0])
        locals_ = [local_tensor(value) for value in values]
        if dim is None:
            full = locals_[0]
            for other in locals_[1:]:
                if other.shape != full.shape or not torch.equal(other, full):
                    raise ValueError(f"Replicated tensor differs across ranks: {key}")
        else:
            full = torch.cat(locals_, dim=dim)
            expected_shape = tuple(values[0].shape)
            if tuple(full.shape) != expected_shape:
                raise ValueError(f"{key}: got {tuple(full.shape)}, expected {expected_shape}")
        full_state[key] = full.contiguous()
        if i % 50 == 0:
            print(f"Consolidated {i}/{len(keys)} tensors", flush=True)

    first = ckpts[0]
    out = {
        "model": full_state,
        "step": first.get("step"),
        "config": first.get("config"),
        "args": first.get("args"),
        "source_sharded_dir": str(sharded_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(out, tmp)
    tmp.replace(output)
    print(f"Wrote {output}", flush=True)


def main() -> None:
    args = parse_args()
    consolidate(Path(args.sharded_dir), Path(args.output))


if __name__ == "__main__":
    main()
