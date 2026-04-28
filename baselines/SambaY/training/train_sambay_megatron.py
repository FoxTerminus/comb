"""Train the SambaY-Llama baseline.

Supports both the original single-process smoke path and the YOCO-style
distributed TP/DP path used for full SambaY training.
"""

from __future__ import annotations

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

from baselines.SambaY.models.SambaY import SambaYConfig, SambaYForCausalLM
from baselines.SambaY.models.SambaY_megatron import apply_tensor_parallelism
from baselines.SambaY.training.data import collate_fn_sambay

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SyntheticSambaYDataset(Dataset):
    """Small deterministic dataset for smoke and overfit validation."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        base = torch.arange(1, seq_len + 1, dtype=torch.long)
        input_ids = ((base - 1) % max(vocab_size - 1, 1)) + 1
        shift_labels = torch.roll(input_ids, shifts=-1)
        shift_labels[-1] = -100
        self.sample = {
            "input_ids": input_ids.tolist(),
            "shift_labels": shift_labels.tolist(),
            "token_length": seq_len,
            "supervised_token_length": int((shift_labels != -100).sum().item()),
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


def parse_args():
    parser = argparse.ArgumentParser(description="SambaY-Llama training")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--init-sambay-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/data3/junhaohu/model/SambaY-Llama-8B")
    parser.add_argument("--ckpt-dir", type=str, default="/data3/junhaohu/checkpoints/SambaY-Llama-8B")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--resume-skip-completed-dataset", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-gpu-0-1", action="store_true")
    parser.add_argument("--tiny-config", action="store_true")
    parser.add_argument("--tiny-hidden-size", type=int, default=32)
    parser.add_argument("--tiny-num-layers", type=int, default=8)
    parser.add_argument("--tiny-vocab-size", type=int, default=128)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-num-samples", type=int, default=64)
    parser.add_argument("--synthetic-seq-len", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--max-steps-per-dataset", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp32-master-optimizer", action="store_true", default=True)
    parser.add_argument("--no-fp32-master-optimizer", action="store_false", dest="fp32_master_optimizer")
    parser.add_argument("--fp32-master-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--max-train-seq-len", type=int, default=0)
    parser.add_argument("--label-shift-mode", choices=["existing", "next-token"], default="existing")
    parser.add_argument("--force-reprocess-data", action="store_true")
    parser.add_argument("--debug-dataset-name", type=str, default=None)
    parser.add_argument("--debug-max-samples", type=int, default=0)
    parser.add_argument("--debug-repeat-factor", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--steps-per-print", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=0)
    return parser.parse_args()


def _check_cuda_visibility(args) -> None:
    if args.allow_gpu_0_1:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        raise RuntimeError(
            "Refusing to use CUDA without CUDA_VISIBLE_DEVICES. "
            "Set CUDA_VISIBLE_DEVICES=5,6 for SambaY tests, or pass --device cpu."
        )
    physical = [item.strip() for item in visible.split(",") if item.strip()]
    if any(item in {"0", "1"} for item in physical):
        raise RuntimeError(
            f"Refusing to use forbidden physical GPUs from CUDA_VISIBLE_DEVICES={visible!r}. "
            "Use CUDA_VISIBLE_DEVICES=5,6."
        )


def resolve_device(args) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        _check_cuda_visibility(args)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        _check_cuda_visibility(args)
        return torch.device("cuda")
    return torch.device("cpu")


def tiny_llama_config(args) -> LlamaConfig:
    num_layers = args.tiny_num_layers
    return LlamaConfig(
        vocab_size=args.tiny_vocab_size,
        hidden_size=args.tiny_hidden_size,
        intermediate_size=args.tiny_hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=max(1, args.tiny_hidden_size // 8),
        num_key_value_heads=max(1, args.tiny_hidden_size // 16),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def build_model(args) -> SambaYForCausalLM:
    if args.init_sambay_path:
        return SambaYForCausalLM.from_pretrained(args.init_sambay_path)
    if args.tiny_config:
        text_config = tiny_llama_config(args)
    else:
        text_config = LlamaConfig.from_pretrained(args.model_name)
    split = text_config.num_hidden_layers // 2
    config = SambaYConfig(
        text_config=text_config,
        num_self_decoder_layers=split,
        num_cross_decoder_layers=text_config.num_hidden_layers - split,
    )
    return SambaYForCausalLM(config)


def load_repo_datasets():
    data_module = importlib.import_module("baselines.SambaY.data")
    return data_module.TRAIN_DATASETS, data_module.DATASET_DICT


def count_supervised_tokens(batch) -> int:
    return int((batch["shift_labels"] != -100).sum().item())


def get_training_loss_log_path(args) -> str:
    log_dir = Path(args.log_dir) if args.log_dir is not None else Path(__file__).resolve().parent
    path = log_dir / "training_loss.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_training_diagnostics_log_path(args) -> str:
    log_dir = Path(args.log_dir) if args.log_dir is not None else Path(__file__).resolve().parent
    path = log_dir / "training_diagnostics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_checkpoint(model, optimizer, scheduler, global_steps: int, dataset_name: str, args) -> str:
    ckpt_path = Path(args.ckpt_dir) / f"step_{global_steps}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_steps": global_steps,
        "dataset_name": dataset_name,
    }
    final_path = ckpt_path / "model.pt"
    tmp_path = ckpt_path / "model.pt.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, final_path)
    return str(ckpt_path)


def load_checkpoint(model, optimizer, scheduler, args, device):
    if args.resume_ckpt is None:
        return 0, None
    ckpt_path = Path(args.resume_ckpt) / "model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["global_steps"]), checkpoint.get("dataset_name")


def build_datasets(args, model) -> tuple[list[str], dict[str, Dataset]]:
    if args.synthetic_data:
        return ["synthetic"], {
            "synthetic": SyntheticSambaYDataset(
                num_samples=args.synthetic_num_samples,
                seq_len=args.synthetic_seq_len,
                vocab_size=model.vocab_size,
            )
        }

    train_datasets, dataset_dict = load_repo_datasets()
    names = [args.debug_dataset_name] if args.debug_dataset_name else train_datasets
    datasets = {}
    for name in names:
        if name not in dataset_dict:
            raise ValueError(f"Unknown SambaY dataset: {name}")
        ds = dataset_dict[name](
            args.model_name,
            split="train_sft" if name == "ultrachat_200k" else "train",
            force_reprocess=args.force_reprocess_data,
        )
        tokenized_ds = ds.data
        if args.debug_max_samples > 0:
            tokenized_ds = RepeatedSubsetDataset(
                tokenized_ds,
                max_samples=args.debug_max_samples,
                repeat_factor=args.debug_repeat_factor,
            )
        datasets[name] = tokenized_ds
    return names, datasets


def build_optimizer(trainable_params, args):
    """Build AdamW, optionally using fp32 master weights for bf16 training."""
    if args.bf16 and args.fp32_master_optimizer:
        master_device = torch.device(args.fp32_master_device)
        master_params = [
            param.detach().float().to(master_device).clone().requires_grad_(True)
            for param in trainable_params
        ]
        optimizer = torch.optim.AdamW(master_params, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer, master_params

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer, None


def optimizer_step(optimizer, trainable_params, master_params) -> None:
    """Step AdamW and sync fp32 master weights back to bf16 model weights."""
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


def set_gradient_checkpointing(model, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = enabled


def clip_grad_norm_tp(parameters, max_norm: float, tp_group, eps: float = 1e-6) -> torch.Tensor:
    """Clip gradients using a norm reduced across tensor-parallel shards."""
    params = [param for param in parameters if param.grad is not None]
    if not params:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        return torch.zeros((), device=device)

    device = params[0].grad.device
    local_sq_norm = torch.zeros((), device=device, dtype=torch.float32)
    for param in params:
        local_sq_norm += param.grad.detach().float().pow(2).sum()

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
    if world_size % tp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size ({tp_size})")

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


def distributed_barrier() -> None:
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def save_distributed_checkpoint(model, global_steps: int, dataset_name: str, args, tp_group, dp_rank) -> None:
    """Save model-only TP shards at dataset boundaries."""
    ckpt_path = Path(args.ckpt_dir) / f"step_{global_steps}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    if dp_rank == 0:
        base_model = model.module if hasattr(model, "module") else model
        tp_rank = dist.get_rank(tp_group)
        checkpoint = {
            "model_state_dict": base_model.state_dict(),
            "global_steps": global_steps,
            "dataset_name": dataset_name,
        }
        final_path = ckpt_path / f"tp_rank_{tp_rank}.pt"
        tmp_path = ckpt_path / f"tp_rank_{tp_rank}.pt.tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    distributed_barrier()


def load_distributed_checkpoint(model, args, tp_group):
    if args.resume_ckpt is None:
        return 0, None

    tp_rank = dist.get_rank(tp_group)
    ckpt_path = Path(args.resume_ckpt) / f"tp_rank_{tp_rank}.pt"
    if not ckpt_path.exists():
        print(f"[Rank {dist.get_rank()}] Checkpoint not found at {ckpt_path}")
        return 0, None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_model = model.module if hasattr(model, "module") else model
    base_model.load_state_dict(checkpoint["model_state_dict"])
    global_steps = int(checkpoint["global_steps"])
    dataset_name = checkpoint.get("dataset_name")

    if dist.get_rank() == 0:
        print("Loaded model-only checkpoint. Optimizer and scheduler state start fresh by design.")
        print(f"Resumed from checkpoint: step={global_steps}, dataset={dataset_name}")
    return global_steps, dataset_name


def main_single_process(args):
    device = resolve_device(args)
    model = build_model(args)
    model.to(device)
    if args.bf16 and device.type == "cuda":
        model = model.bfloat16()
    model.train()

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in trainable_params)
    if total_params != trainable_count:
        raise RuntimeError(f"SambaY should use full-parameter training: {trainable_count} != {total_params}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupDecayLR(optimizer, args.warmup_steps, args.total_steps)
    global_steps, resume_dataset = load_checkpoint(model, optimizer, scheduler, args, device)

    loss_log_path = get_training_loss_log_path(args)
    diagnostics_log_path = get_training_diagnostics_log_path(args)
    if not os.path.exists(diagnostics_log_path):
        with open(diagnostics_log_path, "w", encoding="utf-8") as f:
            f.write("dataset,global_step,loss,lr,grad_norm,total_tokens,supervised_tokens,max_seqlen_q,device\n")

    dataset_names, datasets = build_datasets(args, model)
    if resume_dataset is not None and resume_dataset not in dataset_names:
        print(f"Resume checkpoint dataset {resume_dataset!r} is not in the current dataset list; continuing normally.")

    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable_count}/{total_params}")
    print(f"Datasets: {dataset_names}")

    for dataset_name in dataset_names:
        dataset = datasets[dataset_name]
        loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=partial(
                collate_fn_sambay,
                max_seq_len=args.max_train_seq_len if args.max_train_seq_len > 0 else None,
                label_shift_mode=args.label_shift_mode,
            ),
        )
        dataset_start_step = global_steps
        for micro_step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip) if args.grad_clip > 0 else None
            optimizer.step()
            scheduler.step()
            global_steps += 1

            if global_steps % args.steps_per_print == 0:
                print(f"[{dataset_name}] step {global_steps}, loss={float(loss.item()):.6f}")

            if global_steps % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                grad_norm_value = float(grad_norm.item()) if grad_norm is not None else 0.0
                total_tokens = int(batch["input_ids"].numel())
                supervised_tokens = count_supervised_tokens(batch)
                max_seqlen_q = int(batch["max_seqlen_q"])
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{dataset_name},{global_steps},{float(loss.item())}\n")
                with open(diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{dataset_name},{global_steps},{float(loss.item())},{lr},{grad_norm_value},"
                        f"{total_tokens},{supervised_tokens},{max_seqlen_q},{device.type}\n"
                    )

            if global_steps >= args.total_steps:
                break
            if args.max_steps_per_dataset > 0 and global_steps - dataset_start_step >= args.max_steps_per_dataset:
                break

        ckpt_path = save_checkpoint(model, optimizer, scheduler, global_steps, dataset_name, args)
        print(f"Checkpoint saved: {ckpt_path}")
        if global_steps >= args.total_steps:
            break

    model.save_pretrained(args.output_dir)
    print(f"SambaY training complete. Model saved to {args.output_dir}")


def main_distributed(args):
    if args.device == "cpu":
        raise RuntimeError("Distributed SambaY training requires CUDA/NCCL. Use the single-process path for CPU smoke.")
    _check_cuda_visibility(args)

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

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in trainable_params)
    if total_params != trainable_count:
        raise RuntimeError(f"SambaY should use full-parameter training: {trainable_count} != {total_params}")

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
    if grad_accum_steps < 1 or args.global_batch_size % (args.micro_batch_size * dp_world_size) != 0:
        raise ValueError(
            "global_batch_size must be divisible by micro_batch_size * dp_world_size, "
            f"got global_batch_size={args.global_batch_size}, "
            f"micro_batch_size={args.micro_batch_size}, dp_world_size={dp_world_size}"
        )
    if dist.get_rank() == 0:
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    global_steps, resume_dataset = load_distributed_checkpoint(model, args, tp_group)
    if master_params is not None:
        with torch.no_grad():
            for param, master_param in zip(trainable_params, master_params):
                master_param.copy_(param.detach().float().to(master_param.device))

    loss_log_path = get_training_loss_log_path(args)
    diagnostics_log_path = get_training_diagnostics_log_path(args)
    if dist.get_rank() == 0 and not os.path.exists(diagnostics_log_path):
        with open(diagnostics_log_path, "w", encoding="utf-8") as f:
            f.write(
                "dataset,global_step,loss,lr,grad_norm,total_tokens,"
                "supervised_tokens,max_seqlen_q,optimizer_mode,peak_memory_gb\n"
            )

    if args.synthetic_data:
        datasets_to_train = ["synthetic"]
        dataset_dict = {}
    else:
        train_datasets, dataset_dict = load_repo_datasets()
        datasets_to_train = [args.debug_dataset_name] if args.debug_dataset_name else train_datasets[:]
        unknown_datasets = [name for name in datasets_to_train if name not in dataset_dict]
        if unknown_datasets:
            raise ValueError(f"Unknown SambaY dataset(s): {unknown_datasets}")
        if args.resume_skip_completed_dataset and resume_dataset and resume_dataset in datasets_to_train:
            idx = datasets_to_train.index(resume_dataset) + 1
            datasets_to_train = datasets_to_train[idx:]

    cpu_num = os.cpu_count() or 1
    base_model = model.module if hasattr(model, "module") else model

    for dataset_name in datasets_to_train:
        if global_steps >= args.total_steps:
            break

        if args.synthetic_data:
            tokenized_ds = SyntheticSambaYDataset(
                num_samples=args.synthetic_num_samples,
                seq_len=args.synthetic_seq_len,
                vocab_size=base_model.vocab_size,
            )
        else:
            ds = None
            if dist.get_rank() == 0:
                ds = dataset_dict[dataset_name](
                    args.model_name,
                    split="train_sft" if dataset_name == "ultrachat_200k" else "train",
                    force_reprocess=args.force_reprocess_data,
                )
            distributed_barrier()
            if dist.get_rank() != 0:
                ds = dataset_dict[dataset_name](
                    args.model_name,
                    split="train_sft" if dataset_name == "ultrachat_200k" else "train",
                    force_reprocess=False,
                )
            distributed_barrier()
            tokenized_ds = ds.data
            if args.debug_max_samples > 0:
                tokenized_ds = RepeatedSubsetDataset(
                    tokenized_ds,
                    max_samples=args.debug_max_samples,
                    repeat_factor=args.debug_repeat_factor,
                )

        if args.max_train_seq_len > 0 and dist.get_rank() == 0:
            print(f"[{dataset_name}] Max SambaY training sequence length: {args.max_train_seq_len}")
        if dist.get_rank() == 0:
            print(f"[{dataset_name}] Label shift mode: {args.label_shift_mode}")
            print(f"[{dataset_name}] Data path: SambaY-native decoder-only")

        sampler = torch.utils.data.DistributedSampler(
            tokenized_ds,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=True,
        )

        data_loader = DataLoader(
            tokenized_ds,
            collate_fn=partial(
                collate_fn_sambay,
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

        for batch in data_loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss / grad_accum_steps
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Non-finite SambaY loss on rank {dist.get_rank()}: {float(loss.detach().item())}")

            loss.backward()
            micro_step += 1
            step_loss += float(loss.item())
            step_total_tokens += int(batch["input_ids"].numel())
            step_supervised_tokens += count_supervised_tokens(batch)
            step_max_seqlen_q = max(step_max_seqlen_q, int(batch["max_seqlen_q"]))

            if micro_step < 3:
                loss_tensor = torch.tensor([loss.item()], device=device)
                loss_list = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size(tp_group))]
                dist.all_gather(loss_list, loss_tensor, group=tp_group)
                if dp_rank == 0 and tp_rank == 0:
                    losses = [tensor.item() for tensor in loss_list]
                    if max(losses) - min(losses) > 1e-3:
                        print(
                            f"WARNING: TP ranks have divergent losses: {losses}. "
                            "Data may not be synchronized across TP ranks."
                        )

            if micro_step % args.steps_per_print == 0 and dist.get_rank() == 0:
                actual_loss = float(loss.item()) * grad_accum_steps
                print(
                    f"[{dataset_name}] micro_step {micro_step}/{total_micro_steps}, "
                    f"global_step {global_steps}, loss: {actual_loss:.6f}"
                )

            if micro_step % grad_accum_steps == 0:
                grad_norm = clip_grad_norm_tp(trainable_params, args.grad_clip, tp_group) if args.grad_clip > 0 else None
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
                    if grad_norm is not None and not torch.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite SambaY grad norm on rank {dist.get_rank()}: {grad_norm_value}")
                    optimizer_mode = "fp32-master" if master_params is not None else "direct"
                    average_loss = log_loss_sum / max(1, log_step_count)
                    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(f"[{dataset_name}] step {global_steps}, avg_loss: {average_loss:.6f}")
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{dataset_name},{global_steps},{average_loss}\n")
                    with open(diagnostics_log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{dataset_name},{global_steps},{average_loss},"
                            f"{lr},{grad_norm_value},{log_total_tokens},"
                            f"{log_supervised_tokens},{log_max_seqlen_q},"
                            f"{optimizer_mode},{peak_memory_gb}\n"
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

                if global_steps >= args.total_steps:
                    break
                if args.save_interval > 0 and global_steps % args.save_interval == 0:
                    save_distributed_checkpoint(model, global_steps, dataset_name, args, tp_group, dp_rank)
                    if dist.get_rank() == 0:
                        print(
                            f"Interval checkpoint saved at step {global_steps}: "
                            f"{Path(args.ckpt_dir) / f'step_{global_steps}'}"
                        )
                if (
                    args.max_steps_per_dataset > 0
                    and global_steps - dataset_start_global_steps >= args.max_steps_per_dataset
                ):
                    break

        if micro_step % grad_accum_steps != 0 and global_steps < args.total_steps:
            grad_norm = clip_grad_norm_tp(trainable_params, args.grad_clip, tp_group) if args.grad_clip > 0 else None
            optimizer_step(optimizer, trainable_params, master_params)
            scheduler.step()
            global_steps += 1

            if global_steps % args.log_interval == 0 and dist.get_rank() == 0:
                lr = scheduler.get_last_lr()[0]
                grad_norm_value = float(grad_norm.item()) if grad_norm is not None else 0.0
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite SambaY grad norm on rank {dist.get_rank()}: {grad_norm_value}")
                optimizer_mode = "fp32-master" if master_params is not None else "direct"
                average_loss = step_loss
                peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{dataset_name},{global_steps},{average_loss}\n")
                with open(diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{dataset_name},{global_steps},{average_loss},"
                        f"{lr},{grad_norm_value},{step_total_tokens},"
                        f"{step_supervised_tokens},{step_max_seqlen_q},"
                        f"{optimizer_mode},{peak_memory_gb}\n"
                    )

            if args.save_interval > 0 and global_steps % args.save_interval == 0:
                save_distributed_checkpoint(model, global_steps, dataset_name, args, tp_group, dp_rank)
                if dist.get_rank() == 0:
                    print(
                        f"Interval checkpoint saved at step {global_steps}: "
                        f"{Path(args.ckpt_dir) / f'step_{global_steps}'}"
                    )

        model.eval()
        save_distributed_checkpoint(model, global_steps, dataset_name, args, tp_group, dp_rank)
        if dist.get_rank() == 0:
            print(
                f"Checkpoint saved at step {global_steps}. "
                f"TP shards are stored under {Path(args.ckpt_dir) / f'step_{global_steps}'}"
            )
        distributed_barrier()
        model.train()

    if dist.get_rank() == 0:
        print("SambaY distributed training complete.")

    dist.destroy_process_group()


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or "RANK" in os.environ:
        main_distributed(args)
    else:
        main_single_process(args)


if __name__ == "__main__":
    main()
