"""Train a YOCO-Llama baseline with the repository's distributed training stack.

Stage 6 focuses on the YOCO-specific training path:

- use the YOCO-only collate function
- train all parameters
- support distributed launch
- support TP adaptation via the YOCO-specific Megatron module
"""

import argparse
import datetime
import importlib
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from transformers import LlamaConfig

from baselines.YOCO.models.YOCO import YOCOConfig, YOCOForCausalLM
from baselines.YOCO.models.YOCO_megatron import apply_tensor_parallelism
from baselines.YOCO.training.data import collate_fn_yoco

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SyntheticYOCODataset(Dataset):
    """Small deterministic dataset for YOCO smoke, overfit, and TP validation."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        base = torch.arange(1, seq_len + 1, dtype=torch.long)
        input_ids = ((base - 1) % max(vocab_size - 1, 1)) + 1
        shift_labels = torch.roll(input_ids, shifts=-1)
        shift_labels[-1] = -100
        self.sample = {
            "input_ids": input_ids.tolist(),
            "shift_labels": shift_labels.tolist(),
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sample


class RepeatedSubsetDataset(Dataset):
    """Repeat a small prefix of a dataset for overfit diagnostics."""

    def __init__(self, dataset, max_samples: int, repeat_factor: int):
        self.dataset = dataset
        self.max_samples = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
        self.repeat_factor = max(1, repeat_factor)

    def __len__(self):
        return self.max_samples * self.repeat_factor

    def __getitem__(self, idx):
        return self.dataset[idx % self.max_samples]


def load_repo_datasets():
    yoco_data = importlib.import_module("baselines.YOCO.data")
    return yoco_data.TRAIN_DATASETS, yoco_data.DATASET_DICT, yoco_data.CPU_NUM


def get_training_loss_log_path() -> str:
    log_path = Path(__file__).resolve().parent / "training_loss.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return str(log_path)


def get_training_diagnostics_log_path() -> str:
    log_path = Path(__file__).resolve().parent / "training_diagnostics.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return str(log_path)


def parse_args():
    parser = argparse.ArgumentParser(description="YOCO-Llama DP training")
    parser.add_argument("--tp-size", type=int, default=4,
                        help="Tensor parallel size")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--total-steps", type=int, default=8000000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Run the model in bf16")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16",
                        help="Disable bf16 and keep the model in fp32")
    parser.add_argument("--fp32-master-optimizer", action="store_true", default=True,
                        help="Keep fp32 master weights and Adam states when model weights are bf16")
    parser.add_argument("--no-fp32-master-optimizer", action="store_false", dest="fp32_master_optimizer",
                        help="Use AdamW directly on model parameters")
    parser.add_argument("--fp32-master-device", choices=["cuda", "cpu"], default="cuda",
                        help="Device for fp32 master weights and Adam states")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--init-yoco-path", type=str, default="/data3/junhaohu/model/YOCO-Llama-8B-Init",
                        help="Optional path to a pre-initialized YOCO checkpoint")
    parser.add_argument("--output-dir", type=str, default="/data3/junhaohu/model/YOCO-Llama-8B")
    parser.add_argument("--ckpt-dir", type=str, default="/data3/junhaohu/checkpoints/YOCO-Llama-8B")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N global steps to training_loss.csv")
    parser.add_argument("--steps-per-print", type=int, default=10,
                        help="Print progress every N micro steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable activation checkpointing for YOCO decoder layers")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable activation checkpointing")
    parser.add_argument("--synthetic-data", action="store_true",
                        help="Use a deterministic synthetic dataset for smoke and overfit tests")
    parser.add_argument("--synthetic-num-samples", type=int, default=64)
    parser.add_argument("--synthetic-seq-len", type=int, default=16)
    parser.add_argument("--max-steps-per-dataset", type=int, default=0,
                        help="If > 0, stop after this many global steps on each dataset")
    parser.add_argument("--max-train-seq-len", type=int, default=0,
                        help="If > 0, left-truncate packed YOCO training sequences to this length")
    parser.add_argument("--label-shift-mode", choices=["existing", "next-token"], default="existing",
                        help="Use existing dataset labels or convert them to causal next-token labels")
    parser.add_argument("--force-reprocess-data", action="store_true",
                        help="Ignore cached YOCO-native datasets and rebuild preprocessing")
    parser.add_argument("--debug-dataset-name", type=str, default=None,
                        help="If set, train only this named dataset")
    parser.add_argument("--debug-max-samples", type=int, default=0,
                        help="If > 0, repeat only this many leading samples for overfit diagnostics")
    parser.add_argument("--debug-repeat-factor", type=int, default=1,
                        help="Repeat factor used with --debug-max-samples")
    parser.add_argument("--debug-log-prefix", type=str, default="",
                        help="Prefix added to dataset names in CSV logs for diagnostic runs")
    parser.set_defaults(bf16=True)
    return parser.parse_args()


def build_optimizer(trainable_params, args):
    """Build AdamW, optionally using fp32 master weights for bf16 training."""
    if args.bf16 and args.fp32_master_optimizer:
        master_device = torch.device(args.fp32_master_device)
        master_params = [
            p.detach().float().to(master_device).clone().requires_grad_(True)
            for p in trainable_params
        ]
        optimizer = torch.optim.AdamW(
            master_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return optimizer, master_params

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return optimizer, None


def optimizer_step(optimizer, trainable_params, master_params):
    """Step AdamW and keep bf16 model params synchronized with fp32 masters."""
    if master_params is None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return

    for param, master_param in zip(trainable_params, master_params):
        if param.grad is None:
            master_param.grad = None
        else:
            master_param.grad = param.grad.detach().float().to(
                device=master_param.device,
                non_blocking=True,
            )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        for param, master_param in zip(trainable_params, master_params):
            param.copy_(master_param.to(device=param.device, dtype=param.dtype, non_blocking=True))
            param.grad = None


def count_supervised_tokens(batch) -> int:
    shift_labels = batch.get("shift_labels")
    if shift_labels is None:
        return 0
    return int((shift_labels != -100).sum().item())


def set_gradient_checkpointing(model, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = enabled


def clip_grad_norm_tp(parameters, max_norm: float, tp_group, eps: float = 1e-6) -> torch.Tensor:
    """Clip gradients with a norm computed across all tensor-parallel shards."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        return torch.zeros((), device=device)

    device = params[0].grad.device
    local_sq_norm = torch.zeros((), device=device, dtype=torch.float32)
    for param in params:
        grad = param.grad.detach()
        local_sq_norm += grad.float().pow(2).sum()

    total_sq_norm = local_sq_norm
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        total_sq_norm = local_sq_norm.clone()
        dist.all_reduce(total_sq_norm, op=dist.ReduceOp.SUM, group=tp_group)

    total_norm = torch.sqrt(total_sq_norm)
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for param in params:
            param.grad.detach().mul_(clip_coef.to(param.grad.device))
    return total_norm


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


def save_checkpoint(model, global_steps, dataset_name, args, tp_group, dp_rank):
    """Save one model-only checkpoint at the end of each dataset.

    YOCO full-parameter Adam optimizer states are large. This baseline keeps all
    dataset-boundary checkpoints, but intentionally stores model weights only.
    Resuming from these checkpoints restores model weights and starts optimizer
    and scheduler state fresh.
    """
    ckpt_path = os.path.join(args.ckpt_dir, f"step_{global_steps}")
    os.makedirs(ckpt_path, exist_ok=True)

    if dp_rank == 0:
        base_model = model.module if hasattr(model, "module") else model
        tp_rank = dist.get_rank(tp_group)
        checkpoint = {
            "model_state_dict": base_model.state_dict(),
            "global_steps": global_steps,
            "dataset_name": dataset_name,
        }

        final_path = os.path.join(ckpt_path, f"tp_rank_{tp_rank}.pt")
        tmp_path = final_path + ".tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    dist.barrier()


def load_checkpoint(model, args, tp_group):
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
    if dist.get_rank() == 0:
        print(
            "Loaded model-only checkpoint. "
            "Optimizer and scheduler state start fresh by design."
        )

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
    set_gradient_checkpointing(model, args.gradient_checkpointing)
    if dist.get_rank() == 0:
        state = "enabled" if args.gradient_checkpointing else "disabled"
        print(f"Gradient checkpointing: {state}")
    model = model.to(device)
    if args.bf16:
        model = model.bfloat16()

    if dp_world_size > 1:
        model = DDP(model, device_ids=[local_rank], process_group=dp_group, find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer, master_params = build_optimizer(trainable_params, args)
    scheduler = WarmupDecayLR(optimizer, args.warmup_steps, args.total_steps)
    optimizer.zero_grad(set_to_none=True)
    for param in trainable_params:
        param.grad = None

    if dist.get_rank() == 0:
        param_dtype = trainable_params[0].dtype if trainable_params else None
        optimizer_mode = "fp32-master" if master_params is not None else "direct"
        print(f"Trainable parameter dtype: {param_dtype}, optimizer mode: {optimizer_mode}")

    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * dp_world_size)
    if grad_accum_steps < 1:
        raise ValueError(
            "global_batch_size must be at least micro_batch_size * dp_world_size, "
            f"got global_batch_size={args.global_batch_size}, "
            f"micro_batch_size={args.micro_batch_size}, dp_world_size={dp_world_size}"
        )
    if dist.get_rank() == 0:
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    global_steps, resume_dataset = load_checkpoint(model, args, tp_group)
    if master_params is not None:
        with torch.no_grad():
            for param, master_param in zip(trainable_params, master_params):
                master_param.copy_(param.detach().float().to(master_param.device))
    loss_log_path = get_training_loss_log_path()
    diagnostics_log_path = get_training_diagnostics_log_path()
    if dist.get_rank() == 0 and not os.path.exists(diagnostics_log_path):
        with open(diagnostics_log_path, "w", encoding="utf-8") as f:
            f.write(
                "dataset,global_step,loss,lr,grad_norm,total_tokens,"
                "supervised_tokens,max_seqlen_q,optimizer_mode\n"
            )

    if args.synthetic_data:
        datasets_to_train = ["synthetic"]
        cpu_num = os.cpu_count() or 1
    else:
        TRAIN_DATASETS, DATASET_DICT, cpu_num = load_repo_datasets()
        datasets_to_train = [args.debug_dataset_name] if args.debug_dataset_name else TRAIN_DATASETS[:]
        unknown_datasets = [name for name in datasets_to_train if name not in DATASET_DICT]
        if unknown_datasets:
            raise ValueError(f"Unknown dataset(s): {unknown_datasets}")
        if resume_dataset and resume_dataset in datasets_to_train:
            idx = datasets_to_train.index(resume_dataset) + 1
            datasets_to_train = datasets_to_train[idx:]

    for dataset_name in datasets_to_train:
        if args.synthetic_data:
            vocab_size = model.module.vocab_size if hasattr(model, "module") else model.vocab_size
            tokenized_ds = SyntheticYOCODataset(
                num_samples=args.synthetic_num_samples,
                seq_len=args.synthetic_seq_len,
                vocab_size=vocab_size,
            )
        else:
            ds = DATASET_DICT[dataset_name](
                args.model_name,
                split="train_sft" if dataset_name == "ultrachat_200k" else "train",
                force_reprocess=args.force_reprocess_data,
            )
            tokenized_ds = ds.data
            if args.debug_max_samples > 0:
                tokenized_ds = RepeatedSubsetDataset(
                    tokenized_ds,
                    max_samples=args.debug_max_samples,
                    repeat_factor=args.debug_repeat_factor,
                )

        log_dataset_name = f"{args.debug_log_prefix}{dataset_name}" if args.debug_log_prefix else dataset_name

        if args.max_train_seq_len > 0 and dist.get_rank() == 0:
            print(f"[{dataset_name}] Max YOCO training sequence length: {args.max_train_seq_len}")
        if dist.get_rank() == 0:
            print(f"[{dataset_name}] Label shift mode: {args.label_shift_mode}")
            print(f"[{dataset_name}] Data path: YOCO-native decoder-only")

        sampler = torch.utils.data.DistributedSampler(
            tokenized_ds,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=True,
        )

        data_loader = DataLoader(
            tokenized_ds,
            collate_fn=partial(
                collate_fn_yoco,
                max_seq_len=args.max_train_seq_len if args.max_train_seq_len > 0 else None,
                label_shift_mode=args.label_shift_mode,
            ),
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=max(cpu_num // dist.get_world_size(), 1),
            drop_last=False,
        )

        micro_step = 0
        step_loss = 0.0
        step_total_tokens = 0
        step_supervised_tokens = 0
        step_max_seqlen_q = 0
        log_loss_sum = 0.0
        log_step_count = 0
        log_total_tokens = 0
        log_supervised_tokens = 0
        log_max_seqlen_q = 0
        total_micro_steps = len(data_loader)
        estimated_optimizer_steps = math.ceil(total_micro_steps / grad_accum_steps) if total_micro_steps else 0
        dataset_start_global_steps = global_steps
        if dist.get_rank() == 0:
            print(
                f"[{dataset_name}] Starting training: "
                f"{total_micro_steps} micro steps, "
                f"{estimated_optimizer_steps} optimizer steps"
            )
            if total_micro_steps == 0:
                print(f"[{dataset_name}] WARNING: dataloader is empty; this dataset will be skipped.")
            elif total_micro_steps < grad_accum_steps:
                print(
                    f"[{dataset_name}] WARNING: only {total_micro_steps} micro steps for "
                    f"grad_accum_steps={grad_accum_steps}; one partial optimizer step will be used."
                )

        for step, batch in enumerate(data_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            micro_step += 1
            step_loss += loss.item()
            step_total_tokens += int(batch["input_ids"].numel())
            step_supervised_tokens += count_supervised_tokens(batch)
            step_max_seqlen_q = max(step_max_seqlen_q, int(batch["max_seqlen_q"]))

            if micro_step < 3:
                loss_tensor = torch.tensor([loss.item()], device=device)
                loss_list = [
                    torch.zeros_like(loss_tensor)
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(loss_list, loss_tensor, group=tp_group)
                if dp_rank == 0 and tp_rank == 0:
                    losses = [t.item() for t in loss_list]
                    if max(losses) - min(losses) > 1e-3:
                        print(
                            f"WARNING: TP ranks have divergent losses: {losses}. "
                            "Data may not be synchronized across TP ranks."
                        )

            if micro_step % args.steps_per_print == 0 and dist.get_rank() == 0:
                actual_loss = loss.item() * grad_accum_steps
                print(
                    f"[{dataset_name}] micro_step {micro_step}/{total_micro_steps}, "
                    f"global_step {global_steps}, loss: {actual_loss:.6f}"
                )

            if micro_step % grad_accum_steps == 0:
                grad_norm = None
                if args.grad_clip > 0:
                    grad_norm = clip_grad_norm_tp(trainable_params, args.grad_clip, tp_group)

                optimizer_step(optimizer, trainable_params, master_params)
                scheduler.step()
                global_steps += 1

                log_loss_sum += step_loss
                log_step_count += 1
                log_total_tokens += step_total_tokens
                log_supervised_tokens += step_supervised_tokens
                log_max_seqlen_q = max(log_max_seqlen_q, step_max_seqlen_q)

                if global_steps % args.log_interval == 0 and dist.get_rank() == 0:
                    lr = scheduler.get_last_lr()[0]
                    grad_norm_value = float(grad_norm.item()) if grad_norm is not None else 0.0
                    optimizer_mode = "fp32-master" if master_params is not None else "direct"
                    average_loss = log_loss_sum / max(1, log_step_count)
                    print(f"[{dataset_name}] step {global_steps}, avg_loss: {average_loss:.6f}")
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{log_dataset_name},{global_steps},{average_loss}\n")
                    with open(diagnostics_log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{log_dataset_name},{global_steps},{average_loss},"
                            f"{lr},{grad_norm_value},{log_total_tokens},"
                            f"{log_supervised_tokens},{log_max_seqlen_q},"
                            f"{optimizer_mode}\n"
                        )
                    log_loss_sum = 0.0
                    log_step_count = 0
                    log_total_tokens = 0
                    log_supervised_tokens = 0
                    log_max_seqlen_q = 0

                step_loss = 0.0
                step_total_tokens = 0
                step_supervised_tokens = 0
                step_max_seqlen_q = 0

                if (
                    args.max_steps_per_dataset > 0
                    and global_steps - dataset_start_global_steps >= args.max_steps_per_dataset
                ):
                    break

        if micro_step % grad_accum_steps != 0:
            if args.grad_clip > 0:
                clip_grad_norm_tp(trainable_params, args.grad_clip, tp_group)
            optimizer_step(optimizer, trainable_params, master_params)
            scheduler.step()
            global_steps += 1

        model.eval()
        save_checkpoint(model, global_steps, dataset_name, args, tp_group, dp_rank)

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
