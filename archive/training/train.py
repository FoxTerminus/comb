#!/usr/bin/env python3
"""Distributed training script for Comb (Qwen3 backbone).

Saves resumable FSDP/DTensor checkpoints and optional gathered full
checkpoints that can be exported with ``tools/export_comb_hf.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.prolong_qwen_dataset import ProLongQwenDataset
from models.comb_qwen import CombForConditionalGeneration
from models.config import CombConfig
from transformers import Qwen3Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ctx-len", type=int, default=65536)
    p.add_argument("--target-len", type=int, default=32768)
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
    p.add_argument("--from-scratch", action="store_true", default=True)
    p.add_argument("--no-from-scratch", dest="from_scratch", action="store_false")
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


def load_config(path: str) -> CombConfig:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    cross_layers = raw.pop("cross_attention_layers")
    return CombConfig(text_config=Qwen3Config(**raw), cross_attention_layers=cross_layers)


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


def maybe_checkpoint_layers(model: CombForConditionalGeneration) -> None:
    for i, layer in enumerate(model.chunk_model.layers):
        model.chunk_model.layers[i] = checkpoint_wrapper(layer)
    for i, layer in enumerate(model.language_model.model.decoder_layers):
        model.language_model.model.decoder_layers[i] = checkpoint_wrapper(layer)
    for i, layer in enumerate(model.language_model.model.cross_layers):
        model.language_model.model.cross_layers[i] = checkpoint_wrapper(layer)


def build_model(config: CombConfig, args: argparse.Namespace, mesh) -> CombForConditionalGeneration:
    model = CombForConditionalGeneration(config, from_scratch=args.from_scratch).cuda()
    if args.act_ckpt:
        maybe_checkpoint_layers(model)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.chunk_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    for layer in model.language_model.model.decoder_layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    for layer in model.language_model.model.cross_layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def make_comb_batch(
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = batch["input_ids"].cuda(non_blocking=True)
    needed = args.ctx_len + args.target_len
    if tokens.shape[1] < needed:
        raise ValueError(f"Need {needed} tokens per sample, got {tokens.shape[1]}")
    chunk_ids = tokens[:, : args.ctx_len].contiguous()
    input_ids = tokens[:, args.ctx_len - 1 : args.ctx_len + args.target_len - 1].contiguous()
    shift_labels = tokens[:, args.ctx_len : args.ctx_len + args.target_len].contiguous()
    position_ids_k = torch.arange(args.ctx_len, device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
    position_ids = torch.arange(args.target_len, device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
    return chunk_ids, input_ids, shift_labels, position_ids, position_ids_k


def make_varlen_metadata(
    batch_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, q_len, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * k_len, k_len, device=device, dtype=torch.int32)
    cu_seqlens_chunk = cu_seqlens_k
    return cu_seqlens_q, cu_seqlens_k, cu_seqlens_chunk, q_len, k_len, k_len


def save_checkpoint(model, optimizer, step: int, config: CombConfig, args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / f"step_{step:06d}_sharded"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "config": config.to_dict(),
            "args": vars(args),
        },
        shard_dir / f"rank_{rank:05d}.pt",
    )
    if is_rank0():
        (shard_dir / "metadata.json").write_text(
            f'{{"step": {step}, "world_size": {dist.get_world_size()}}}\n',
            encoding="utf-8",
        )
    barrier()
    print0(f"Saved sharded checkpoint dir: {shard_dir}")


def save_full_checkpoint(model, step: int, config: CombConfig, args: argparse.Namespace) -> Path | None:
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
                "config": config.to_dict(),
                "args": vars(args),
            },
            tmp_path,
        )
        tmp_path.replace(path)
        print0(f"Saved full checkpoint: {path}")
    barrier()
    return path


def maybe_export_hf(full_path: Path | None, args: argparse.Namespace) -> None:
    if not args.export_hf_final or not is_rank0() or full_path is None:
        return
    from tools.export_comb_hf import export_checkpoint

    out = args.hf_output_dir or str(Path("/data3/junhaohu/model") / Path(args.output_dir).name)
    export_checkpoint(
        ckpt_path=str(full_path),
        output=out,
        config_path=args.config,
        max_shard_size=args.max_shard_size,
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

    if args.resume and args.from_scratch:
        print0("--resume supplied; using --no-from-scratch behavior before loading checkpoint")
        args.from_scratch = False

    config = load_config(args.config)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    model = build_model(config, args, mesh)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=0.1,
    )
    start_step = load_resume(model, optimizer, args.resume) if args.resume else 0

    dataset = ProLongQwenDataset(args.data_dir, block_size=args.ctx_len + args.target_len, seed=args.seed)
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
            chunk_ids, input_ids, shift_labels, position_ids, position_ids_k = make_comb_batch(batch, args)
            (
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_chunk,
                max_seqlen_q,
                max_seqlen_k,
                max_seqlen_chunk,
            ) = make_varlen_metadata(input_ids.shape[0], args.target_len, args.ctx_len, input_ids.device)
            loss_accum = 0.0
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(
                        chunk_ids=chunk_ids,
                        input_ids=input_ids,
                        shift_labels=shift_labels,
                        position_ids=position_ids,
                        position_ids_k=position_ids_k,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        cu_seqlens_chunk=cu_seqlens_chunk,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        max_seqlen_chunk=max_seqlen_chunk,
                    ).loss / args.grad_accum
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
    maybe_export_hf(full_path, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
