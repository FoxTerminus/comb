"""Train a YOCO-Llama baseline with the repository's distributed training stack.

Stage 6 focuses on the YOCO-specific training path:

- use the YOCO-only collate function
- train all parameters
- support distributed launch
- support TP adaptation via the YOCO-specific Megatron module
"""

import argparse
import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from transformers import LlamaConfig

from baselines.YOCO.models.YOCO import YOCOConfig, YOCOForCausalLM
from baselines.YOCO.models.YOCO_megatron import apply_tensor_parallelism
from baselines.YOCO.training.data import collate_fn_yoco
from data import TRAIN_DATASETS, DATASET_DICT
from data.base import CPU_NUM


def parse_args():
    parser = argparse.ArgumentParser(description="YOCO-Llama DP training")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--total-steps", type=int, default=8000000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--init-yoco-path", type=str, default=None,
                        help="Optional path to a pre-initialized YOCO checkpoint")
    parser.add_argument("--output-dir", type=str, default="/data3/junhaohu/model/YOCO-Llama-8B-Init")
    parser.add_argument("--ckpt-dir", type=str, default="/data3/junhaohu/checkpoints/YOCO-Llama-8B-Init")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N global steps to training_loss.csv")
    parser.add_argument("--steps-per-print", type=int, default=10,
                        help="Print progress every N micro steps")
    return parser.parse_args()


def initialize_parallel_groups(tp_size: int):
    world_size = dist.get_world_size()

    assert world_size % tp_size == 0, (
        f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
    )
    dp_size = world_size // tp_size
    rank = dist.get_rank()

    tp_group = None
    for i in range(dp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

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
            factor = max(
                0.0,
                (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps),
            )
        return [base_lr * factor for base_lr in self.base_lrs]


def save_checkpoint(model, optimizer, scheduler, global_steps, dataset_name, args, tp_group, dp_rank):
    ckpt_path = os.path.join(args.ckpt_dir, f"step_{global_steps}")
    os.makedirs(ckpt_path, exist_ok=True)

    if dp_rank == 0:
        base_model = model.module if hasattr(model, "module") else model
        tp_rank = dist.get_rank(tp_group)
        torch.save(
            {
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_steps": global_steps,
                "dataset_name": dataset_name,
            },
            os.path.join(ckpt_path, f"tp_rank_{tp_rank}.pt"),
        )

    dist.barrier()


def load_checkpoint(model, optimizer, scheduler, args, tp_group):
    if args.resume_ckpt is None:
        return 0, None

    tp_rank = dist.get_rank(tp_group)
    ckpt_path = os.path.join(args.resume_ckpt, f"tp_rank_{tp_rank}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[Rank {dist.get_rank()}] Checkpoint not found at {ckpt_path}")
        return 0, None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_model = model.module if hasattr(model, "module") else model
    base_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    global_steps = checkpoint["global_steps"]
    dataset_name = checkpoint["dataset_name"]
    if dist.get_rank() == 0:
        print(f"Resumed from checkpoint: step={global_steps}, dataset={dataset_name}")

    return global_steps, dataset_name


def build_model(args):
    if args.init_yoco_path is not None:
        model = YOCOForCausalLM.from_pretrained(args.init_yoco_path)
    else:
        text_config = LlamaConfig.from_pretrained(args.model_name)
        config = YOCOConfig(text_config=text_config, num_self_decoder_layers=16, num_cross_decoder_layers=16)
        model = YOCOForCausalLM(config)
    return model


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tp_group, dp_group, tp_rank, dp_rank, dp_world_size = initialize_parallel_groups(args.tp_size)

    if dist.get_rank() == 0:
        print(
            f"World size: {dist.get_world_size()}, TP size: {args.tp_size}, "
            f"DP size: {dp_world_size}"
        )

    model = build_model(args)
    apply_tensor_parallelism(model, tp_group)
    model = model.to(device)
    if args.bf16:
        model = model.bfloat16()

    if dp_world_size > 1:
        model = DDP(model, device_ids=[local_rank], process_group=dp_group, find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupDecayLR(optimizer, args.warmup_steps, args.total_steps)

    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * dp_world_size)
    if dist.get_rank() == 0:
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    global_steps, resume_dataset = load_checkpoint(model, optimizer, scheduler, args, tp_group)

    datasets_to_train = TRAIN_DATASETS[:]
    if resume_dataset and resume_dataset in datasets_to_train:
        idx = datasets_to_train.index(resume_dataset) + 1
        datasets_to_train = datasets_to_train[idx:]

    for dataset_name in datasets_to_train:
        ds = DATASET_DICT[dataset_name](
            args.model_name,
            split="train_sft" if dataset_name == "ultrachat_200k" else "train",
        )
        tokenized_ds = ds.data

        sampler = torch.utils.data.DistributedSampler(
            tokenized_ds,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=True,
        )

        data_loader = DataLoader(
            tokenized_ds,
            collate_fn=collate_fn_yoco,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=max(CPU_NUM // dist.get_world_size(), 1),
            drop_last=False,
        )

        micro_step = 0
        accumulated_loss = 0.0
        total_micro_steps = len(data_loader)
        if dist.get_rank() == 0:
            print(
                f"[{dataset_name}] Starting training: "
                f"{total_micro_steps} micro steps, "
                f"{total_micro_steps // grad_accum_steps} global steps"
            )

        for step, batch in enumerate(data_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            micro_step += 1
            accumulated_loss += loss.item()

            if micro_step % args.steps_per_print == 0 and dist.get_rank() == 0:
                actual_loss = loss.item() * grad_accum_steps
                print(
                    f"[{dataset_name}] micro_step {micro_step}/{total_micro_steps}, "
                    f"global_step {global_steps}, loss: {actual_loss:.6f}"
                )

            if micro_step % grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

                if global_steps % args.log_interval == 0 and dist.get_rank() == 0:
                    print(f"[{dataset_name}] step {global_steps}, loss: {accumulated_loss:.6f}")
                    with open("training_loss.csv", "a", encoding="utf-8") as f:
                        f.write(f"{dataset_name},{global_steps},{accumulated_loss}\n")

                accumulated_loss = 0.0

        if micro_step % grad_accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_steps += 1

        model.eval()
        save_checkpoint(model, optimizer, scheduler, global_steps, dataset_name, args, tp_group, dp_rank)

        if dist.get_rank() == 0:
            print(
                f"Checkpoint saved at step {global_steps}. "
                f"TP shards are stored under {os.path.join(args.ckpt_dir, f'step_{global_steps}')}"
            )

        dist.barrier()
        model.train()

    if dist.get_rank() == 0:
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
