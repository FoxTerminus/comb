"""Preflight checks before launching SambaY TP4 full training.

This script intentionally avoids loading the full 9.93B model. It checks the
environment, visible GPU policy, dependency imports, checkpoint layout, and
writable training directories before a long run.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="SambaY TP4 full-training preflight")
    parser.add_argument("--init-sambay-path", default="/data3/junhaohu/model/SambaY-Llama-8B-Init")
    parser.add_argument("--ckpt-dir", default="/data3/junhaohu/checkpoints/SambaY-Llama-8B")
    parser.add_argument("--log-dir", default="/data3/junhaohu/comb/baselines/SambaY/training")
    parser.add_argument("--expected-visible-gpus", type=int, default=4)
    parser.add_argument("--expected-cuda-visible-devices", default="2,3,4,7")
    parser.add_argument("--allow-gpu-0-1", action="store_true")
    return parser.parse_args()


def check_cuda_visibility(args) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set.")

    physical = [item.strip() for item in visible.split(",") if item.strip()]
    expected_physical = [
        item.strip()
        for item in args.expected_cuda_visible_devices.split(",")
        if item.strip()
    ]
    if expected_physical and physical != expected_physical:
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={args.expected_cuda_visible_devices}, got {visible}"
        )
    if len(physical) != args.expected_visible_gpus:
        raise RuntimeError(
            f"Expected {args.expected_visible_gpus} visible GPUs, got {len(physical)}: {visible}"
        )
    if not args.allow_gpu_0_1 and any(item in {"0", "1"} for item in physical):
        raise RuntimeError(f"Refusing forbidden physical GPU0/GPU1: CUDA_VISIBLE_DEVICES={visible}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false.")
    if torch.cuda.device_count() != args.expected_visible_gpus:
        raise RuntimeError(
            f"torch sees {torch.cuda.device_count()} GPUs, expected {args.expected_visible_gpus}."
        )


def check_imports() -> None:
    for module_name in ("mamba_ssm", "causal_conv1d"):
        importlib.import_module(module_name)
    importlib.import_module("baselines.SambaY.models.SambaY")
    importlib.import_module("baselines.SambaY.models.SambaY_megatron")
    data_module = importlib.import_module("baselines.SambaY.data")
    if not getattr(data_module, "TRAIN_DATASETS", None):
        raise RuntimeError("baselines.SambaY.data.TRAIN_DATASETS is empty.")


def check_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Init SambaY checkpoint does not exist: {path}")
    required = ["config.json"]
    missing = [name for name in required if not (path / name).exists()]
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    if missing or not has_weights:
        raise RuntimeError(
            f"Invalid init checkpoint layout at {path}. missing={missing}, has_weights={has_weights}"
        )


def check_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".sambay_preflight_write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()


def main() -> None:
    args = parse_args()
    check_cuda_visibility(args)
    check_imports()
    check_checkpoint(Path(args.init_sambay_path))
    check_writable_dir(Path(args.ckpt_dir))
    check_writable_dir(Path(args.log_dir))
    print("SambaY TP4 preflight passed.")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"init_sambay_path={args.init_sambay_path}")
    print(f"ckpt_dir={args.ckpt_dir}")
    print(f"log_dir={args.log_dir}")


if __name__ == "__main__":
    main()
