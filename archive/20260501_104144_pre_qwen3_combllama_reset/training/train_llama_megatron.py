"""Train a CombLlama model using Tensor Parallel (TP) + Data Parallel (DP).

Launch with torchrun:
    torchrun --nproc_per_node=4 train_llama_megatron.py

With 4 GPUs and tp-size=4: TP groups = {0,1,2,3}, DP groups = {0}, {1}, {2}, and {3}.
"""

import argparse
import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from transformers import LlamaConfig

from models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
from models.CombLlama_megatron import apply_tensor_parallelism
from data import TRAIN_DATASETS, DATASET_DICT
from data.base import CPU_NUM, collate_fn


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CombLlama Megatron DP+TP Training")
    parser.add_argument("--tp-size", type=int, default=4,
                        help="Tensor parallel size (must divide num_attention_heads and num_kv_heads)")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--total-steps", type=int, default=8000000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=str, default="/data3/junhaohu/model/CombLlama-8B-Instruct")
    parser.add_argument("--ckpt-dir", type=str, default="/data3/junhaohu/checkpoints/CombLlama-8B-Instruct")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N global steps to training_loss.csv")
    parser.add_argument("--steps-per-print", type=int, default=10,
                        help="Print progress every N micro steps")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parallel Group Initialization
# ---------------------------------------------------------------------------

def initialize_parallel_groups(tp_size: int):
    """Initialize TP and DP process groups.

    With world_size GPUs and tp_size, we create:
    - TP groups: consecutive ranks of size tp_size (e.g., {0,1}, {2,3} for tp=2, world=4)
    - DP groups: ranks at same position within their TP group (e.g., {0,2}, {1,3})

    Returns:
        (tp_group, dp_group, tp_rank, dp_rank)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % tp_size == 0, (
        f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    )
    dp_size = world_size // tp_size

    # TP groups: [0, 1, ..., tp-1], [tp, tp+1, ..., 2*tp-1], ...
    tp_group = None
    for i in range(dp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    # DP groups: [0, tp, 2*tp, ...], [1, tp+1, 2*tp+1, ...], ...
    dp_group = None
    for i in range(tp_size):
        ranks = list(range(i, world_size, tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            dp_group = group

    tp_rank = dist.get_rank(tp_group)
    dp_rank = dist.get_rank(dp_group)
    dp_world_size = dist.get_world_size(dp_group)

    return tp_group, dp_group, tp_rank, dp_rank, dp_world_size


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------

class WarmupDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by linear decay."""

    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            factor = max(0.0, (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps))
        return [base_lr * factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, global_steps, dataset_name, args,
                    tp_group, dp_rank):
    """Save checkpoint: full model weights (gathered from TP) + optimizer state."""
    ckpt_path = os.path.join(args.ckpt_dir, f"step_{global_steps}")
    os.makedirs(ckpt_path, exist_ok=True)

    # Save full model weights by gathering TP shards on dp_rank=0
    if dp_rank == 0:
        tp_rank = dist.get_rank(tp_group)
        # Each TP rank saves its shard
        shard_path = os.path.join(ckpt_path, f"tp_rank_{tp_rank}.pt")
        # Get the base model (unwrap DDP)
        base_model = model.module if hasattr(model, "module") else model
        torch.save({
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_steps": global_steps,
            "dataset_name": dataset_name,
            "tp_size": dist.get_world_size(tp_group),
        }, shard_path)

    dist.barrier()


def load_checkpoint(model, optimizer, scheduler, args, tp_group, dp_rank):
    """Load checkpoint and return (global_steps, dataset_name) or (0, None)."""
    if args.resume_ckpt is None:
        return 0, None

    tp_rank = dist.get_rank(tp_group)
    shard_path = os.path.join(args.resume_ckpt, f"tp_rank_{tp_rank}.pt")

    if not os.path.exists(shard_path):
        print(f"[Rank {dist.get_rank()}] Checkpoint not found at {shard_path}")
        return 0, None

    checkpoint = torch.load(shard_path, map_location="cpu", weights_only=False)
    base_model = model.module if hasattr(model, "module") else model
    base_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    global_steps = checkpoint["global_steps"]
    dataset_name = checkpoint["dataset_name"]

    if dist.get_rank() == 0:
        print(f"Resumed from checkpoint: step={global_steps}, dataset={dataset_name}")

    return global_steps, dataset_name


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Initialize distributed
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tp_group, dp_group, tp_rank, dp_rank, dp_world_size = initialize_parallel_groups(args.tp_size)

    if dist.get_rank() == 0:
        print(f"World size: {dist.get_world_size()}, TP size: {args.tp_size}, "
              f"DP size: {dp_world_size}")

    # 2. Build model
    model = CombLlamaForConditionalGeneration(
        from_scratch=True,
        config=CombLlamaConfig(LlamaConfig.from_pretrained(args.model_name)),
    )

    # 3. Apply tensor parallelism (splits weights across TP group)
    apply_tensor_parallelism(model, tp_group)
    model = model.to(device)

    if args.bf16:
        model = model.bfloat16()

    # 4. Freeze parameters (same logic as DeepSpeed script)
    for param in model.language_model.parameters():
        param.requires_grad = False
    for param in model.chunk_model.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.language_model.model.cross_layers.parameters():
        param.requires_grad = True

    # 5. Wrap with DDP for data parallelism (using DP process group)
    if dp_world_size > 1:
        model = DDP(model, device_ids=[local_rank], process_group=dp_group,
                    find_unused_parameters=True)

    base_model = model.module if hasattr(model, "module") else model

    # 6. Optimizer and scheduler (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = WarmupDecayLR(optimizer, args.warmup_steps, args.total_steps)

    # Gradient accumulation
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * dp_world_size)
    if dist.get_rank() == 0:
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    # 7. Load checkpoint if resuming
    global_steps, resume_dataset = load_checkpoint(
        model, optimizer, scheduler, args, tp_group, dp_rank,
    )

    # Re-establish requires_grad after checkpoint loading
    for param in base_model.language_model.parameters():
        param.requires_grad = False
    for param in base_model.chunk_model.embed_tokens.parameters():
        param.requires_grad = False
    for param in base_model.language_model.model.cross_layers.parameters():
        param.requires_grad = True

    # Skip completed datasets
    datasets_to_train = TRAIN_DATASETS[:]
    if resume_dataset and resume_dataset in datasets_to_train:
        idx = datasets_to_train.index(resume_dataset) + 1
        datasets_to_train = datasets_to_train[idx:]

    # 8. Training loop
    scaler = None  # BF16 does not need GradScaler

    for dataset_name in datasets_to_train:
        ds = DATASET_DICT[dataset_name](args.model_name, split="train_sft" if dataset_name == "ultrachat_200k" else "train")
        tokenized_ds = ds.data

        # DP-aware sampler: TP ranks in same group get same data
        sampler = torch.utils.data.DistributedSampler(
            tokenized_ds,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=True,
        )

        data_loader = DataLoader(
            tokenized_ds,
            collate_fn=collate_fn,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=max(CPU_NUM // dist.get_world_size(), 1),
            drop_last=False,
        )

        micro_step = 0
        accumulated_loss = 0.0
        total_micro_steps = len(data_loader)
        if dist.get_rank() == 0:
            print(f"[{dataset_name}] Starting training: "
                  f"{total_micro_steps} micro steps, "
                  f"{total_micro_steps // grad_accum_steps} global steps")

        for step, batch in enumerate(data_loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            # Verify TP data consistency (first few steps only)
            if micro_step < 3:
                loss_tensor = torch.tensor([loss.item()], device=device)
                loss_list = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(loss_list, loss_tensor)
                if dist.get_rank() == 0:
                    losses = [t.item() for t in loss_list]
                    if max(losses) - min(losses) > 1e-3:
                        print(f"WARNING: TP ranks have divergent losses: {losses}. "
                              f"Data may not be synchronized across TP ranks!")
                    else:
                        print(f"TP loss check passed: {losses}")

            # Backward
            loss.backward()
            micro_step += 1
            accumulated_loss += loss.item()

            # Per-micro-step progress
            if micro_step % args.steps_per_print == 0 and dist.get_rank() == 0:
                actual_loss = loss.item() * grad_accum_steps
                print(f"[{dataset_name}] micro_step {micro_step}/{total_micro_steps}, "
                      f"global_step {global_steps}, loss: {actual_loss:.6f}")

            # Optimizer step at accumulation boundary
            if micro_step % grad_accum_steps == 0:
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

                # Logging
                if global_steps % args.log_interval == 0 and dist.get_rank() == 0:
                    print(f"[{dataset_name}] step {global_steps}, loss: {accumulated_loss:.6f}")
                    with open("training_loss.csv", "a") as f:
                        f.write(f"{dataset_name},{global_steps},{accumulated_loss}\n")

                accumulated_loss = 0.0

        # Handle remaining micro-steps that didn't reach accumulation boundary
        if micro_step % grad_accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_steps += 1

        # Save after each dataset
        model.eval()
        save_checkpoint(model, optimizer, scheduler, global_steps, dataset_name, args,
                        tp_group, dp_rank)

        # Also save full HF model (gather TP shards) on rank 0
        if dist.get_rank() == 0:
            output_dir = f"{args.output_dir}({loss.item() * grad_accum_steps:.6f})"
            # Note: for full HF save, use checkpoint_converter.py to merge TP shards
            print(f"Checkpoint saved at step {global_steps}. "
                  f"Use checkpoint_converter.py to convert to HF format.")

        dist.barrier()
        model.train()

    if dist.get_rank() == 0:
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
