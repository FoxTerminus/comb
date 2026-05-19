#!/usr/bin/env python3
"""Distributed training script for SambaY / SambaYOCO.

This entrypoint saves both the usual resumable FSDP/DTensor checkpoint and,
when requested, a gathered full checkpoint suitable for evaluation/export.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.ArchScale.data.prolong_dataset import ProLongPackedDataset
from baselines.ArchScale.models.config import Config
from baselines.ArchScale.models.model import GPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sambay_d16")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
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
    p.add_argument("--act-ckpt", action="store_true")
    p.add_argument("--diagnostics-file", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save-full-final", action="store_true", default=True)
    p.add_argument("--no-save-full-final", dest="save_full_final", action="store_false")
    p.add_argument("--save-full-interval", type=int, default=0)
    p.add_argument("--export-hf-final", action="store_true")
    p.add_argument("--hf-output-dir", default=None)
    p.add_argument("--max-shard-size", default="5GB")
    return p.parse_args()


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs) -> None:
    if is_rank0():
        print(*args, **kwargs, flush=True)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(arg: str) -> Config:
    path = Path(arg)
    return Config.from_yaml(str(path)) if path.exists() else Config.from_name(arg)


def lr_for_step(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        mult = step / max(1, args.warmup_steps)
    else:
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        mult = args.min_lr_mult + (1.0 - args.min_lr_mult) * cosine
    return args.lr * mult


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_model(config: Config, act_ckpt: bool, mesh) -> GPT:
    model = GPT(config).cuda()
    if act_ckpt:
        for i, block in enumerate(model.blocks):
            model.blocks[i] = checkpoint_wrapper(block)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def save_checkpoint(model, optimizer, step: int, config: Config, args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / f"step_{step:06d}_sharded"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "config_snapshot": asdict(config),
        "args": vars(args),
    }
    torch.save(ckpt, shard_dir / f"rank_{rank:05d}.pt")
    if is_rank0():
        (shard_dir / "metadata.json").write_text(
            f'{{"step": {step}, "world_size": {dist.get_world_size()}}}\n',
            encoding="utf-8",
        )
    barrier()
    print0(f"Saved sharded checkpoint dir: {shard_dir}")


def save_full_checkpoint(model, step: int, config: Config, args: argparse.Namespace) -> Path | None:
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(model, options=opts)
    path = None
    if is_rank0():
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"step_{step:06d}_full.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": full_sd,
                "step": step,
                "config_snapshot": asdict(config),
                "args": vars(args),
            },
            tmp_path,
        )
        tmp_path.replace(path)
        print0(f"Saved full checkpoint: {path}")
    barrier()
    return path


def maybe_export_hf(full_path: Path | None, config: Config, args: argparse.Namespace) -> None:
    if not args.export_hf_final or not is_rank0() or full_path is None:
        return
    from tools.export_samba_hf import export_checkpoint

    out = args.hf_output_dir or str(Path("/data3/junhaohu/model") / Path(args.output_dir).name)
    export_checkpoint(
        ckpt_path=str(full_path),
        output=out,
        config_arg=args.config,
        max_shard_size=args.max_shard_size,
        validate_load=False,
    )


def load_resume(model, optimizer, path: str) -> int:
    p = Path(path)
    if p.is_dir():
        p = p / f"rank_{dist.get_rank():05d}.pt"
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print0(f"Resumed from {p} at step {ckpt.get('step', 0)}")
    return int(ckpt.get("step", 0))


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    seed_all(args.seed + dist.get_rank())

    config = load_config(args.config)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    model = build_model(config, args.act_ckpt, mesh)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=0.1,
    )

    start_step = load_resume(model, optimizer, args.resume) if args.resume else 0
    dataset = ProLongPackedDataset(args.data_dir, block_size=args.ctx_len, seed=args.seed)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.micro_bsz, sampler=sampler, num_workers=0, pin_memory=True)
    print0(dataset.summary())

    diag_path = Path(args.diagnostics_file) if args.diagnostics_file else Path(args.output_dir) / "training_diagnostics.csv"
    if is_rank0() and start_step == 0:
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("w", newline="") as f:
            csv.writer(f).writerow(["global_step", "loss", "lr"])

    step = start_step
    model.train()
    while step < args.total_steps:
        sampler.set_epoch(step)
        for batch in loader:
            if step >= args.total_steps:
                break
            lr = lr_for_step(step + 1, args)
            set_lr(optimizer, lr)
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            loss_accum = 0.0
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, labels=labels)["loss"] / args.grad_accum
                loss.backward()
                loss_accum += float(loss.detach()) * args.grad_accum
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1

            loss_tensor = torch.tensor(loss_accum, device="cuda")
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_val = float(loss_tensor.item())
            if is_rank0() and step % args.log_interval == 0:
                print(f"step={step} loss={loss_val:.4f} lr={lr:.6g}", flush=True)
                with diag_path.open("a", newline="") as f:
                    csv.writer(f).writerow([step, loss_val, lr])
            if step % args.save_interval == 0:
                save_checkpoint(model, optimizer, step, config, args)
            if args.save_full_interval and step % args.save_full_interval == 0:
                save_full_checkpoint(model, step, config, args)

    save_checkpoint(model, optimizer, step, config, args)
    full_path = save_full_checkpoint(model, step, config, args) if args.save_full_final else None
    maybe_export_hf(full_path, config, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
