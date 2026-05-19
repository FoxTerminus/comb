#!/usr/bin/env python3
"""Export a full Comb checkpoint to a HuggingFace-style safetensors directory.

Important: this script expects a consolidated/full state_dict.  The FSDP2
training checkpoints produced by saving a DTensor state_dict from one rank are
not enough; they contain only local shards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from transformers import Qwen3Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.config import CombConfig
from models.comb_qwen import CombForConditionalGeneration


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "README.md",
    "LICENSE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Full .pt checkpoint with key `model`, or a raw state_dict.")
    p.add_argument("--config", default="/data3/junhaohu/comb/configs/comb_qwen_1b.yaml")
    p.add_argument("--output", default="/data3/junhaohu/model/Comb-Qwen3-1B-Prolong")
    p.add_argument("--tokenizer-src", default="/data3/junhaohu/model/Qwen3-0.6B")
    p.add_argument("--max-shard-size", default="5GB")
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
                "You need to save a full state_dict from the distributed training job first."
            )
    return state_dict


def load_config(path: str) -> CombConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cross_layers = raw.pop("cross_attention_layers")
    return CombConfig(
        text_config=Qwen3Config(**raw),
        cross_attention_layers=cross_layers,
    )


def copy_tokenizer_files(src: Path, dst: Path) -> None:
    for name in TOKENIZER_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


def export_checkpoint(
    ckpt_path: str,
    output: str,
    config_path: str = "/data3/junhaohu/comb/configs/comb_qwen_1b.yaml",
    tokenizer_src: str = "/data3/junhaohu/model/Qwen3-0.6B",
    max_shard_size: str = "5GB",
) -> None:
    ckpt_path = Path(ckpt_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = require_full_state_dict(state_dict, str(ckpt_path))

    model = CombForConditionalGeneration(load_config(config_path), from_scratch=False)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(
        out,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    copy_tokenizer_files(Path(tokenizer_src), out)

    export_info = {
        "source_checkpoint": str(ckpt_path),
        "source_step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "format": "huggingface_safetensors",
        "max_shard_size": max_shard_size,
    }
    (out / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")
    print(f"Exported {out}")


def main() -> None:
    args = parse_args()
    export_checkpoint(
        ckpt_path=args.ckpt,
        output=args.output,
        config_path=args.config,
        tokenizer_src=args.tokenizer_src,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
