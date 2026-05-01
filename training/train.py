#!/usr/bin/env python3
"""Distributed training script for Comb (Qwen3-0.6B backbone).

Adapted from ``baselines/ArchScale/training/train.py`` which was
validated on 1/2/4 GPU synthetic + 64K preflight.

Usage (4 GPUs, FSDP2)::

    CUDA_VISIBLE_DEVICES=2,3,4,7 torchrun --nnodes=1 --nproc_per_node=4 \\
        comb/training/train.py \\
        --config /data3/junhaohu/comb/configs/comb_qwen_1b.yaml \\
        --data-dir /data3/junhaohu/data/prolong_qwen_v2_subset \\
        --ctx-len 65536 --total-steps 5913 --warmup-steps 591 --lr 3e-4 \\
        --act-ckpt
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import Qwen3Config
from models.config import CombConfig
from models.comb_qwen import CombForConditionalGeneration
from data.prolong_qwen_dataset import ProLongQwenDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_gpu_guard(world_size: int):
    """Stage-based GPU validation.  Single=6, 2-card=5/6, 4-card=2/3/4/7."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set")
    gpus = [int(x.strip()) for x in visible.split(",")]
    if world_size == 1:
        allowed = {6}
    elif world_size == 2:
        allowed = {5, 6}
    elif world_size == 4:
        allowed = {0, 1, 2, 3, 4, 5, 6, 7}
    else:
        raise RuntimeError(f"Unsupported world_size={world_size}. Use 1, 2, or 4.")
    for g in gpus:
        if g not in allowed:
            raise RuntimeError(f"GPU {g} not allowed for world_size={world_size}. Allowed: {allowed}")
    if len(gpus) < world_size:
        raise RuntimeError(f"Need {world_size} GPUs, only {len(gpus)} visible: {visible}")


def _scalar(val):
    if isinstance(val, torch.Tensor):
        if hasattr(val, "full_tensor"):
            val = val.full_tensor()
        return float(val.item())
    return float(val)


def cosine_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _load_config(name):
    if name.endswith((".yaml", ".yml")) or os.path.isfile(name):
        import yaml
        with open(name) as f:
            data = yaml.safe_load(f)
        txt_data = {k: v for k, v in data.items() if k not in ("cross_attention_layers", "chunk_token_index", "pad_token_id", "tie_word_embeddings")}
        cross_layers = data.get("cross_attention_layers", None)
        txt_cfg = Qwen3Config(**txt_data) if txt_data else Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
        return CombConfig(text_config=txt_cfg, cross_attention_layers=cross_layers)
    txt_cfg = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    return CombConfig(text_config=txt_cfg)


def save_checkpoint(model, optimizer, step, config, args, rank, path):
    model_sd = model.state_dict()
    opt_sd = optimizer.state_dict()
    state = {
        "model": model_sd, "optimizer": opt_sd, "step": step,
        "rng_states": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().cpu(),
            "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()],
        },
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        "args": vars(args),
    }
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)


def load_checkpoint(model, optimizer, path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng_states", {})
    if "python" in rng: random.setstate(rng["python"])
    if "numpy" in rng: np.random.set_state(rng["numpy"])
    if "torch" in rng: torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
    if "torch_cuda" in rng:
        tc = [t.cpu() if hasattr(t, "cpu") else t for t in rng["torch_cuda"]]
        torch.cuda.set_rng_state_all(tc)
    return state["step"]


def _find_latest_checkpoint(output_dir):
    candidates = sorted(glob.glob(os.path.join(output_dir, "step_*.pt")), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _check_gpu_guard(world_size)  # after dist init, before model loading

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        print(f"World: {world_size}, GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES','?')}, seed: {args.seed}")

    # --- Model ---
    config = _load_config(args.config)
    config.text_config.max_position_embeddings = args.ctx_len

    model = CombForConditionalGeneration(config, from_scratch=args.from_scratch)
    mesh = init_device_mesh("cuda", (world_size,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # Per-Block FSDP2 for memory efficiency at 64K
    from torch.distributed._composable.fsdp import CPUOffloadPolicy
    cpu_offload = CPUOffloadPolicy(pin_memory=True)
    fully_shard(model, mesh=mesh, mp_policy=mp, offload_policy=cpu_offload)

    if args.act_ckpt:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper, apply_activation_checkpointing,
        )
        from models.comb_qwen import ChunkVarlenLayer, CombCrossAttentionDecoderLayer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, (Qwen3DecoderLayer, ChunkVarlenLayer, CombCrossAttentionDecoderLayer)),
        )

    if rank == 0:
        total = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Model: {total:.1f}M total, {trainable:.1f}M trainable")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    # --- Data: sliding windows ---
    window_len = args.ctx_len + args.target_len
    stride = args.target_len  # each step advances by target_len
    full_ds = ProLongQwenDataset(
        data_dir=args.data_dir, block_size=window_len, seed=args.seed,
        stride=stride,
    )
    rank_indices = list(range(rank, len(full_ds), world_size))
    rank_subset = torch.utils.data.Subset(full_ds, rank_indices)
    loader = DataLoader(rank_subset, batch_size=args.micro_bsz)
    prefix_len = args.ctx_len
    target_len = args.target_len

    if rank == 0:
        print(full_ds.summary())
        total_windows = full_ds.total_blocks
        per_rank = total_windows // world_size
        budget = per_rank * target_len * world_size
        print(f"Sliding window: {window_len} tokens (prefix={prefix_len}, target={target_len}, stride={stride})")
        print(f"Total windows: {total_windows}, per rank: {per_rank}, target budget: {budget/1e9:.2f}B tokens")
        print(f"Required steps: {args.total_steps}, available per rank: {per_rank}")
        # Init diagnostics CSV
        csv_path = os.path.join(args.output_dir, "training_diagnostics.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "loss", "lr"])

    # --- Resume ---
    global_step = 0
    resume_path = args.resume
    if args.resume == "auto":
        resume_path = _find_latest_checkpoint(args.output_dir)
    if resume_path and os.path.isfile(resume_path):
        global_step = load_checkpoint(model, optimizer, resume_path, device)
        if rank == 0: print(f"Resumed from step {global_step}")

    # --- Training loop ---
    model.train()
    total_loss = 0.0
    total_tokens_seen = 0
    micro_step = 0
    t0 = time.time()

    try:
        for batch in loader:
            micro_step += 1
            full_block = batch["input_ids"].to(device)
            assert full_block.shape[0] == 1
            total_T = full_block.shape[1]
            assert total_T == prefix_len + target_len, f"{total_T} != {prefix_len}+{target_len}"
            # Prefix → chunk encoder
            prefix = full_block[:, :prefix_len]
            n_chunks = 64
            assert prefix_len % n_chunks == 0, f"prefix_len {prefix_len} not divisible by {n_chunks}"
            chunk_size = prefix_len // n_chunks
            chunk_ids = prefix.view(1, n_chunks, chunk_size).reshape(1, -1)
            chunk_pos_ids = torch.arange(0, chunk_size, device=device).unsqueeze(0).repeat(1, n_chunks)
            cu_seqlens_chunk = torch.arange(0, prefix_len + 1, chunk_size, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, prefix_len], dtype=torch.int32, device=device)

            # Target → decoder (causal LM)
            target = full_block[:, prefix_len:]
            cu_seqlens_q = torch.tensor([0, target_len], dtype=torch.int32, device=device)
            sl = target.clone()
            sl[:, :-1] = target[:, 1:]
            sl[:, -1] = -100

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=target, chunk_ids=chunk_ids, shift_labels=sl,
                    position_ids=torch.arange(0, target_len, device=device).unsqueeze(0),
                    position_ids_k=chunk_pos_ids,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    cu_seqlens_chunk=cu_seqlens_chunk,
                    max_seqlen_q=target_len,
                    max_seqlen_k=prefix_len,
                    max_seqlen_chunk=chunk_size,
                )
                loss = out.loss / args.grad_accum

            loss.backward()
            total_loss += loss.item() * args.grad_accum
            total_tokens_seen += target.numel()

            if micro_step % args.grad_accum == 0:
                global_step += 1
                lr = cosine_lr(global_step, args.warmup_steps, args.total_steps, args.lr, args.lr * args.min_lr_mult)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step_loss = total_loss / args.grad_accum
                if rank == 0:
                    with open(os.path.join(args.output_dir, "training_diagnostics.csv"), "a", newline="") as f:
                        csv.writer(f).writerow([global_step, step_loss, lr])
                if rank == 0 and global_step % args.log_interval == 0:
                    elapsed = time.time() - t0
                    tk_per_sec = total_tokens_seen / max(elapsed, 1e-8)
                    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                    print(f"step {global_step:5d}/{args.total_steps} | loss: {step_loss:.4f} | lr: {lr:.2e} | tok/s: {tk_per_sec:.0f} | mem: {peak_gb:.1f}GB", flush=True)
                total_loss = 0.0; total_tokens_seen = 0; t0 = time.time()

                if global_step % args.save_interval == 0:
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
                    save_checkpoint(model, optimizer, global_step, config, args, rank, ckpt_path)
                    if rank == 0: print(f"Checkpoint: {ckpt_path}", flush=True)

                if global_step >= args.total_steps:
                    break
        if global_step >= args.total_steps:
            pass
        # Final checkpoint (if not already saved on this step)
        if global_step % args.save_interval != 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
            save_checkpoint(model, optimizer, global_step, config, args, rank, ckpt_path)
            if rank == 0:
                print(f"Final checkpoint: {ckpt_path}", flush=True)

    finally:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Comb training")
    p.add_argument("--config", type=str, default="comb_qwen_1b")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./checkpoints")
    p.add_argument("--ctx-len", type=int, default=65536)
    p.add_argument("--target-len", type=int, default=65536,
                   help="Decoder target length (prefix=ctx-len, total block=ctx-len+target-len)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-mult", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=591)
    p.add_argument("--total-steps", type=int, default=5913)
    p.add_argument("--micro-bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--act-ckpt", action="store_true")
    p.add_argument("--from-scratch", action="store_true", default=True,
                   help="Load Qwen3-0.6B pretrained weights and init Comb (default)")
    p.add_argument("--no-from-scratch", action="store_false", dest="from_scratch",
                   help="Random init (for tiny smoke tests)")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
