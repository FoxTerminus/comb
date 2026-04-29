"""Distributed YOCO gradient spike diagnostic.

This script is intentionally read-only with respect to checkpoints: it loads TP
shards, runs a few real-data forward/backward passes, reports pre-clip gradient
norms and the largest local gradient tensors, and exits without optimizer steps
or checkpoint writes.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from baselines.YOCO.models.YOCO_megatron import apply_tensor_parallelism
from baselines.YOCO.training.data import collate_fn_yoco
from baselines.YOCO.training.train_yoco_megatron import (
    build_model,
    initialize_parallel_groups,
    load_checkpoint,
    load_repo_datasets,
    set_gradient_checkpointing,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose YOCO ChatsV2 gradients")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--init-yoco-path", default="/data3/junhaohu/model/YOCO-Llama-8B-Init")
    parser.add_argument("--resume-ckpt", default="/data3/junhaohu/checkpoints/YOCO-Llama-8B/step_1624")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset-name", default="chats_v2_k200000")
    parser.add_argument("--max-train-seq-len", type=int, default=3072)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--label-shift-mode", choices=["existing", "next-token"], default="existing")
    parser.add_argument("--force-reprocess-data", action="store_true")
    return parser.parse_args()


def check_cuda_visible() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "2,3,4,7":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=2,3,4,7, got {visible!r}")
    if any(item.strip() in {"0", "1"} for item in visible.split(",")):
        raise RuntimeError(f"Forbidden GPU visible in CUDA_VISIBLE_DEVICES={visible!r}")


def grad_norm_and_top(model, tp_group, top_k: int = 8):
    local_sq = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    top = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grad_f = grad.float()
        local_sq += grad_f.pow(2).sum()
        max_abs = float(grad_f.abs().max().item())
        rms = float(torch.sqrt(grad_f.pow(2).mean()).item())
        top.append((max_abs, rms, name, tuple(param.shape)))
    total_sq = local_sq.clone()
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=tp_group)
    total_norm = torch.sqrt(total_sq)
    top.sort(reverse=True, key=lambda item: item[0])
    return float(total_norm.item()), top[:top_k]


def main():
    args = parse_args()
    check_cuda_visible()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    tp_group, _dp_group, tp_rank, dp_rank, dp_world_size = initialize_parallel_groups(args.tp_size)
    if dp_world_size != 1:
        raise RuntimeError("This diagnostic expects pure TP with DP size 1.")

    model = build_model(args)
    apply_tensor_parallelism(model, tp_group)
    model.to(device)
    if args.bf16:
        model.bfloat16()
    set_gradient_checkpointing(model, args.gradient_checkpointing)
    load_checkpoint(model, args, tp_group)
    model.train()
    model.zero_grad(set_to_none=True)

    train_datasets, dataset_dict, cpu_num = load_repo_datasets()
    if args.dataset_name not in dataset_dict:
        raise ValueError(f"Unknown dataset {args.dataset_name}; available={train_datasets}")
    ds = dataset_dict[args.dataset_name](
        args.model_name,
        split="train_sft" if args.dataset_name == "ultrachat_200k" else "train",
        force_reprocess=args.force_reprocess_data,
    )
    sampler = None
    if args.shuffle:
        sampler = torch.utils.data.DistributedSampler(
            ds.data,
            num_replicas=1,
            rank=0,
            shuffle=True,
        )
    loader = DataLoader(
        ds.data,
        collate_fn=partial(
            collate_fn_yoco,
            max_seq_len=args.max_train_seq_len,
            label_shift_mode=args.label_shift_mode,
        ),
        batch_size=args.micro_batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=max(cpu_num // dist.get_world_size(), 1),
        drop_last=False,
    )

    if dist.get_rank() == 0:
        print(
            f"diagnostic dataset={args.dataset_name} max_seq_len={args.max_train_seq_len} shuffle={args.shuffle} "
            f"steps={args.steps} grad_accum_steps={args.grad_accum_steps} "
            f"gradient_checkpointing={args.gradient_checkpointing} ckpt={args.resume_ckpt}"
        )

    accum_loss = 0.0
    accum_tokens = 0
    accum_supervised = 0
    accum_max_q = 0
    for step, batch in enumerate(loader, start=1):
        if step > args.steps:
            break
        if (step - 1) % args.grad_accum_steps == 0:
            model.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_tokens = 0
            accum_supervised = 0
            accum_max_q = 0
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss / args.grad_accum_steps
        loss.backward()
        accum_loss += float(outputs.loss.item())
        accum_tokens += int(batch["input_ids"].numel())
        accum_supervised += int((batch["shift_labels"] != -100).sum().item())
        accum_max_q = max(accum_max_q, int(batch["max_seqlen_q"]))
        if step % args.grad_accum_steps != 0 and step < args.steps:
            continue
        total_norm, top = grad_norm_and_top(model, tp_group)
        supervised = int((batch["shift_labels"] != -100).sum().item())
        if tp_rank == 0 and dp_rank == 0:
            print(
                f"step={step} avg_loss={accum_loss / max(1, min(args.grad_accum_steps, step)):.6f} "
                f"grad_norm={total_norm:.6e} tokens={accum_tokens} supervised={accum_supervised} "
                f"max_seqlen_q={accum_max_q} last_supervised={supervised}"
            )
            for max_abs, rms, name, shape in top:
                print(f"  top_grad max_abs={max_abs:.6e} rms={rms:.6e} name={name} shape={shape}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
