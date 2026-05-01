#!/usr/bin/env python3
"""Distributed training script for SambaY / Samba+YOCO.

Uses PyTorch FSDP2 (``fully_shard``) with bf16 autocast and
cosine learning-rate schedule.

Usage (4 GPUs, pure FSDP2 data-parallel)::

    CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nnodes=1 --nproc_per_node=4 \\
        baselines/ArchScale/training/train.py \\
        --config sambay_d16 \\
        --data-dir /data3/junhaohu/data/prolong_64K_v2_subset \\
        --output-dir /data3/junhaohu/model/SambaY-1B-Prolong \\
        --total-steps 5913 --warmup-steps 591 --lr 3e-4

Config can be a preset name (``sambay_d16``) or a YAML path.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.ArchScale.models.config import Config
from baselines.ArchScale.models.model import Block, GPT
from baselines.ArchScale.data.prolong_dataset import ProLongPackedDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scalar(val: torch.Tensor | float) -> float:
    """Convert a (potentially DTensor) tensor to a Python float."""
    if isinstance(val, torch.Tensor):
        # FSDP2 may return a DTensor — use .full_tensor() if available
        if hasattr(val, "full_tensor"):
            val = val.full_tensor()
        return float(val.item())
    return float(val)


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------

FORBIDDEN_PHYSICAL_GPUS = {0, 1, 5}


def _check_gpu_guard() -> None:
    """Refuse to run if any forbidden physical GPU is visible."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be set explicitly. "
            f"Physical GPUs {FORBIDDEN_PHYSICAL_GPUS} are forbidden."
        )
    for item in visible.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            gpu = int(item)
        except ValueError:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES contains non-integer value {item!r}"
            )
        if gpu in FORBIDDEN_PHYSICAL_GPUS:
            raise RuntimeError(
                f"Physical GPU {gpu} is forbidden. "
                f"Allowed: all except {FORBIDDEN_PHYSICAL_GPUS}. "
                f"Current CUDA_VISIBLE_DEVICES={visible!r}"
            )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(name: str) -> Config:
    """Load a Config from a preset name or YAML file path."""
    if name.endswith((".yaml", ".yml")) or os.path.isfile(name):
        return Config.from_yaml(name)
    return Config.from_name(name)


# ---------------------------------------------------------------------------
# Cosine LR scheduler
# ---------------------------------------------------------------------------

def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Config,
    args: argparse.Namespace,
    rank: int,
    path: str,
) -> None:
    """Save a full training checkpoint.

    All ranks participate in ``state_dict()`` (FSDP2 collective), but
    only rank 0 writes to disk.  The caller should ensure all ranks
    enter this function simultaneously.
    """
    # FSDP2 collectives — must be called on all ranks together
    model_sd = model.state_dict()
    opt_sd = optimizer.state_dict()

    state = {
        "model": model_sd,
        "optimizer": opt_sd,
        "step": step,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().cpu(),
            "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()],
        },
        "config_snapshot": {
            k: v for k, v in config.__dict__.items() if not k.startswith("_")
        },
        "args": vars(args),
    }
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    """Load a training checkpoint (rank 0 loads, then all sync FSDP2 state)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng_states", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch_rng = rng["torch"]
        if hasattr(torch_rng, "cpu"):
            torch_rng = torch_rng.cpu()
        torch.set_rng_state(torch_rng)
    if "torch_cuda" in rng:
        tc = rng["torch_cuda"]
        tc = [t.cpu() if hasattr(t, "cpu") else t for t in tc]
        torch.cuda.set_rng_state_all(tc)
    return state["step"]


def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the most recent ``step_*.pt``, or None."""
    candidates = sorted(
        glob.glob(os.path.join(output_dir, "step_*.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _init_diagnostics(path: str, resume: bool) -> None:
    """Create the rank-0 training diagnostics CSV if needed."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if resume and os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "loss", "lr"])


def _append_diagnostics(
    path: str,
    global_step: int,
    loss: float,
    lr: float,
) -> None:
    """Append one optimizer-step record for loss curve comparison."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([global_step, loss, lr])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # -- distributed init ------------------------------------------------
    _check_gpu_guard()
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # -- RNG seeds -------------------------------------------------------
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        print(f"World size: {world_size}, local_rank: {local_rank}")
        print(f"Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")
        print(f"Seed: {args.seed}")

    # -- model -----------------------------------------------------------
    config = _load_config(args.config)
    config.block_size = args.ctx_len

    # Create model on CPU first (no meta-device), then FSDP2
    model = GPT(config)
    model.reset_parameters()

    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    # Single top-level shard — per-Block fully_shard caused backward
    # hang with 2+ ranks in some PyTorch 2.6 FSDP2 configurations.
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    # Activation checkpointing for 64K context
    if args.act_ckpt:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, Block),
        )
        if rank == 0:
            print("Activation checkpointing: enabled")

    if rank == 0:
        total_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model: {total_m:.1f}M params, FSDP2 ready")

    # -- optimizer -------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )

    # -- data ------------------------------------------------------------
    full_dataset = ProLongPackedDataset(
        data_dir=args.data_dir,
        block_size=args.ctx_len,
        seed=args.seed,
    )
    # Pre-load all blocks into a TensorDataset (avoids Subset/DataLoader
    # iterator hangs observed in 2-GPU configs).
    all_ids = [full_dataset[i]["input_ids"] for i in range(len(full_dataset))]
    all_labels = [full_dataset[i]["labels"] for i in range(len(full_dataset))]
    ids_tensor = torch.stack(all_ids)
    labels_tensor = torch.stack(all_labels)
    rank_indices = list(range(rank, len(full_dataset), world_size))
    loader = DataLoader(
        TensorDataset(ids_tensor[rank_indices], labels_tensor[rank_indices]),
        batch_size=args.micro_bsz,
    )
    if rank == 0:
        print(full_dataset.summary())
        print(f"Tokens/step: {world_size * args.micro_bsz * args.ctx_len:,}")
        print(f"Total steps: {args.total_steps}, warmup: {args.warmup_steps}")

    diagnostics_path = args.diagnostics_file
    if diagnostics_path is None:
        diagnostics_path = os.path.join(args.output_dir, "training_diagnostics.csv")
    if rank == 0:
        _init_diagnostics(diagnostics_path, resume=bool(args.resume))

    # -- resume ----------------------------------------------------------
    global_step = 0
    resume_path = args.resume
    if args.resume == "auto":
        resume_path = _find_latest_checkpoint(args.output_dir)
    if resume_path and os.path.isfile(resume_path):
        global_step = load_checkpoint(model, optimizer, resume_path, device)
        if rank == 0:
            print(f"Resumed from step {global_step} ({resume_path})")

    # -- training loop ---------------------------------------------------
    model.train()
    total_loss = 0.0
    total_tokens_seen = 0
    micro_step = 0
    t0 = time.time()

    try:
        for batch in loader:
            micro_step += 1
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            total_loss += loss.item() * args.grad_accum
            total_tokens_seen += input_ids.numel()

            if micro_step % args.grad_accum == 0:
                global_step += 1


                lr = cosine_lr(
                    global_step, args.warmup_steps, args.total_steps,
                    args.lr, args.lr * args.min_lr_mult,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if rank == 0 and global_step % args.log_interval == 0:
                    avg_loss = total_loss / (args.log_interval * args.grad_accum)
                    elapsed = time.time() - t0
                    tk_per_sec = total_tokens_seen / max(elapsed, 1e-8)
                    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                    print(
                        f"step {global_step:5d}/{args.total_steps} | "
                        f"loss: {avg_loss:.4f} | "
                        f"lr: {lr:.2e} | "
                        
                        f"tok/s: {tk_per_sec:.0f} | "
                        f"mem: {peak_gb:.1f}GB",
                        flush=True,
                    )
                    _append_diagnostics(diagnostics_path, global_step, avg_loss, lr)
                    total_loss = 0.0
                    total_tokens_seen = 0
                    t0 = time.time()

                if global_step % args.save_interval == 0:
                    ckpt_path = os.path.join(
                        args.output_dir, f"step_{global_step:06d}.pt"
                    )
                    save_checkpoint(model, optimizer, global_step, config,
                                    args, rank, ckpt_path)
                    if rank == 0:
                        print(f"Checkpoint saved: {ckpt_path}", flush=True)

                if global_step >= args.total_steps:
                    break

        # final checkpoint
        final_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
        save_checkpoint(model, optimizer, global_step, config, args, rank, final_path)
        if rank == 0:
            print(f"Training complete. Final checkpoint: {final_path}")

    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SambaY / Samba+YOCO training")
    p.add_argument("--config", type=str, required=True,
                   help="Model config (preset name e.g. sambay_d16, or YAML path)")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing ProLong .bin files")
    p.add_argument("--output-dir", type=str, default="./checkpoints",
                   help="Output directory for checkpoints")
    p.add_argument("--ctx-len", type=int, default=65536)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-mult", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=591)
    p.add_argument("--total-steps", type=int, default=5913)
    p.add_argument("--micro-bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--act-ckpt", action="store_true",
                   help="Enable activation checkpointing (required for 64K context)")
    p.add_argument("--diagnostics-file", type=str, default=None,
                   help="Rank-0 CSV path for global_step,loss,lr")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path, or 'auto' to auto-detect latest in --output-dir")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
