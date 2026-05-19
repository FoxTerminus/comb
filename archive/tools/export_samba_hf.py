#!/usr/bin/env python3
"""Export a full SambaY/SambaYOCO checkpoint to a HF-like safetensors dir.

SambaY is not a ``transformers.PreTrainedModel`` in this repo, so the export is
"HF-like": ``config.json`` plus sharded ``model*.safetensors`` and
``model.safetensors.index.json``.  The tensors keep the native module key names
used by :class:`baselines.ArchScale.models.model.GPT`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.ArchScale.models.config import Config
from baselines.ArchScale.models.model import GPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Full .pt checkpoint with key `model`, or raw state_dict.")
    p.add_argument("--config", default=None, help="Preset name or YAML path. Optional if ckpt has config_snapshot.")
    p.add_argument("--output", required=True)
    p.add_argument("--max-shard-size", default="5GB")
    p.add_argument("--validate-load", action="store_true")
    return p.parse_args()


def require_full_state_dict(state_dict: dict, ckpt_path: str) -> dict:
    for key, value in state_dict.items():
        if hasattr(value, "to_local") or value.__class__.__name__ == "DTensor":
            local_shape = tuple(value.to_local().shape) if hasattr(value, "to_local") else "unknown"
            global_shape = tuple(value.shape) if hasattr(value, "shape") else "unknown"
            raise RuntimeError(
                "This checkpoint is a DTensor/FSDP shard, not a full checkpoint.\n"
                f"Checkpoint: {ckpt_path}\n"
                f"First sharded tensor: {key}, global_shape={global_shape}, local_shape={local_shape}\n"
                "Re-save with --save-full-final or --save-full-interval first."
            )
    return state_dict


def load_config(ckpt: dict, config_arg: str | None) -> Config:
    if isinstance(ckpt, dict) and ckpt.get("config_snapshot"):
        return Config(**ckpt["config_snapshot"])
    if not config_arg:
        raise ValueError("--config is required when checkpoint has no config_snapshot")
    path = Path(config_arg)
    return Config.from_yaml(str(path)) if path.exists() else Config.from_name(config_arg)


def write_safetensors_dir(state_dict: dict, config: Config, output: Path, max_shard_size: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {name: state_dict[name].contiguous() for name in tensor_names}
        save_file(shard, output / filename, metadata={"format": "pt"})

    (output / "config.json").write_text(
        json.dumps({"model_type": "sambay", **asdict(config)}, indent=2),
        encoding="utf-8",
    )
    if split.is_sharded:
        index = {
            "metadata": split.metadata,
            "weight_map": split.tensor_to_filename,
        }
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2),
            encoding="utf-8",
        )


def export_checkpoint(
    ckpt_path: str,
    output: str,
    config_arg: str | None = None,
    max_shard_size: str = "5GB",
    validate_load: bool = False,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = require_full_state_dict(state_dict, ckpt_path)
    config = load_config(ckpt, config_arg)
    if validate_load:
        model = GPT(config)
        model.load_state_dict(state_dict, strict=True)
        del model
    out = Path(output)
    write_safetensors_dir(state_dict, config, out, max_shard_size)
    info = {
        "source_checkpoint": ckpt_path,
        "source_step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "format": "hf_like_safetensors",
        "max_shard_size": max_shard_size,
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Exported {out}")


def main() -> None:
    args = parse_args()
    export_checkpoint(
        ckpt_path=args.ckpt,
        output=args.output,
        config_arg=args.config,
        max_shard_size=args.max_shard_size,
        validate_load=args.validate_load,
    )


if __name__ == "__main__":
    main()
