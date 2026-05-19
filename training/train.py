#!/usr/bin/env python3
"""Pure tensor-parallel training script for packed-FA CombLlama."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import LlamaConfig

from comb.data.base import DEFAULT_LABEL_KEY, collate_fn
from comb.models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
from comb.models.tensor_parallel import apply_tensor_parallelism
from comb.models.tp_checkpoint import merge_state_dict_from_tp


DEFAULT_TRAIN_DATASETS = ["SQuAD", "Natural-Instructions", "XSum", "Super-Natural-Instructions"]
DEFAULT_OUTPUT_DIR = "/data3/junhaohu/checkpoints/CombLlama"
DEFAULT_VALIDATION_SIZES = {
    "SQuAD": 2048,
    "XSum": 2048,
    "Natural-Instructions": 8192,
    "Super-Natural-Instructions": 4096,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="CombLlama config JSON/YAML path. If omitted, use --text-model.")
    parser.add_argument("--text-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--data",
        nargs="+",
        default=None,
        help=(
            "One or more local distillation dataset directories saved by datasets.save_to_disk(). "
            "If omitted, train over the default Comb datasets in order."
        ),
    )
    parser.add_argument("--dataset-name", default=None, help="Name written to Comb-style training_loss.csv for single-dataset runs.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY)
    parser.add_argument(
        "--from-scratch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize language model from --text-model weights.",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint dir or rank checkpoint file.")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Optional optimizer-step limit. If omitted, train through all datasets exactly once.",
    )
    parser.add_argument("--micro-batch-size", type=int, default=64, help="Maximum samples per packed micro-batch.")
    parser.add_argument("--max-text-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--max-chunk-tokens-per-batch", type=int, default=32768)
    parser.add_argument("--max-text-len", type=int, default=2048)
    parser.add_argument("--max-chunk-len", type=int, default=8192)
    parser.add_argument("--length-bucket-size", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--lr-schedule-steps",
        type=int,
        default=8_000_000,
        help="Optimizer-step horizon used only for lr decay. This does not limit actual training steps.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=5,
        help="Keep only the latest N step_* checkpoint directories. Use <=0 to keep all checkpoints.",
    )
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-size", type=int, default=None, help="Override validation examples per dataset.")
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=0, help="Optional cap per validation dataset; 0 evaluates all.")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-full-final", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*args, **kwargs) -> None:
    if is_rank0():
        print(*args, **kwargs, flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(args: argparse.Namespace) -> CombLlamaConfig:
    if args.config is None:
        return CombLlamaConfig(text_config=LlamaConfig.from_pretrained(args.text_model))

    path = Path(args.config)
    with path.open(encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    if "text_config" in raw:
        return CombLlamaConfig(**raw)

    raw = dict(raw)
    cross_attention_layers = raw.pop("cross_attention_layers", None)
    chunk_token_index = raw.pop("chunk_token_index", 128255)
    num_hidden_layers = raw.get("num_hidden_layers", 40)
    pad_token_id = raw.get("pad_token_id", 128004)
    tie_word_embeddings = raw.get("tie_word_embeddings", False)
    return CombLlamaConfig(
        text_config=LlamaConfig(**raw),
        cross_attention_layers=cross_attention_layers,
        chunk_token_index=chunk_token_index,
        num_hidden_layers=num_hidden_layers,
        pad_token_id=pad_token_id,
        tie_word_embeddings=tie_word_embeddings,
    )


def default_dataset_paths(text_model: str) -> list[Path]:
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    model_suffix = text_model.replace("/", "_")
    return [hf_home / "datasets" / f"{name}_{model_suffix}" for name in DEFAULT_TRAIN_DATASETS]


def load_dataset(path: Path) -> Dataset:
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Dataset path must be a local directory saved by datasets.save_to_disk(): {str(path)!r}")

    try:
        from datasets import load_from_disk

        loaded = load_from_disk(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to load dataset from {str(path)!r}") from exc
    if isinstance(loaded, dict):
        if "train" not in loaded:
            raise ValueError(f"DatasetDict loaded from {str(path)!r} must contain a `train` split.")
        loaded = loaded["train"]
    return loaded


def infer_dataset_name(path: Path, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    name = path.name
    return name.split("_", 1)[0] if "_" in name else name


def load_datasets(args: argparse.Namespace) -> list[tuple[str, Dataset]]:
    paths = [Path(item) for item in args.data] if args.data else default_dataset_paths(args.text_model)
    if args.dataset_name and len(paths) != 1:
        raise ValueError("--dataset-name can only be used when exactly one --data path is provided.")

    datasets = []
    for path in paths:
        name = infer_dataset_name(path, args.dataset_name if len(paths) == 1 else None)
        dataset = load_dataset(path)
        datasets.append((name, dataset))
    return datasets


def validation_size_for_dataset(name: str, dataset: Dataset, args: argparse.Namespace) -> int:
    if not args.validation:
        return 0
    requested = args.validation_size
    if requested is None:
        requested = DEFAULT_VALIDATION_SIZES.get(name, 2048)
    if requested <= 0:
        return 0
    return min(requested, max(0, len(dataset) - 1))


def split_train_validation(
    datasets: list[tuple[str, Dataset]], args: argparse.Namespace
) -> tuple[list[tuple[str, Dataset]], list[tuple[str, Dataset]]]:
    train_datasets: list[tuple[str, Dataset]] = []
    validation_datasets: list[tuple[str, Dataset]] = []
    for offset, (name, dataset) in enumerate(datasets):
        validation_size = validation_size_for_dataset(name, dataset, args)
        if validation_size == 0:
            train_datasets.append((name, dataset))
            continue

        rng = np.random.default_rng(args.validation_seed + offset)
        indices = rng.permutation(len(dataset))
        validation_indices = np.sort(indices[:validation_size])
        train_indices = np.sort(indices[validation_size:])
        train_dataset = dataset.select(train_indices)
        validation_dataset = dataset.select(validation_indices)
        train_datasets.append((name, train_dataset))
        validation_datasets.append((name, validation_dataset))
    return train_datasets, validation_datasets


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _column_lengths(dataset: Dataset, column: str) -> np.ndarray:
    if not hasattr(dataset, "data"):
        return np.asarray([len(dataset[index][column]) for index in range(len(dataset))], dtype=np.int32)

    try:
        import pyarrow.compute as pc

        lengths = pc.list_value_length(dataset.data.column(column))
        lengths = lengths.combine_chunks()
        indices = getattr(dataset, "_indices", None)
        if indices is not None:
            lengths = pc.take(lengths, indices.column(0).combine_chunks())
        return lengths.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    except Exception:
        return np.asarray([len(dataset[index][column]) for index in range(len(dataset))], dtype=np.int32)


class PackedBatchSampler(Sampler[list[int]]):
    """Build packed batches under sample-count and token-count budgets."""

    def __init__(self, dataset: Dataset, args: argparse.Namespace, epoch: int) -> None:
        if args.label_key not in dataset.column_names:
            raise KeyError(f"`label_key={args.label_key}` is not present in the dataset.")
        self.dataset = dataset
        self.args = args
        self.epoch = epoch
        input_lens = _column_lengths(dataset, "input_ids")
        label_lens = _column_lengths(dataset, args.label_key)
        self.text_lens = input_lens + label_lens
        self.chunk_lens = _column_lengths(dataset, "chunk_ids")

        valid = (self.text_lens <= args.max_text_len) & (self.chunk_lens <= args.max_chunk_len)
        self.valid_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
        self.filtered_count = int(len(dataset) - len(self.valid_indices))
        self._num_batches: int | None = None

    def _ordered_indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.args.seed + self.epoch)
        indices = self.valid_indices.copy()
        rng.shuffle(indices)

        bucket_size = self.args.length_bucket_size
        if bucket_size > 1:
            total_lens = self.text_lens + self.chunk_lens
            for start in range(0, len(indices), bucket_size):
                end = min(start + bucket_size, len(indices))
                bucket = indices[start:end]
                order = np.argsort(total_lens[bucket], kind="stable")
                indices[start:end] = bucket[order]
        return indices

    def _iter_batches(self):
        batch: list[int] = []
        text_tokens = 0
        chunk_tokens = 0
        for index in self._ordered_indices():
            idx = int(index)
            text_len = int(self.text_lens[idx])
            chunk_len = int(self.chunk_lens[idx])
            would_exceed = (
                len(batch) >= self.args.micro_batch_size
                or text_tokens + text_len > self.args.max_text_tokens_per_batch
                or chunk_tokens + chunk_len > self.args.max_chunk_tokens_per_batch
            )
            if batch and would_exceed:
                yield batch
                batch = []
                text_tokens = 0
                chunk_tokens = 0

            batch.append(idx)
            text_tokens += text_len
            chunk_tokens += chunk_len

        if batch:
            yield batch

    def __iter__(self):
        return self._iter_batches()

    def __len__(self) -> int:
        if self._num_batches is None:
            self._num_batches = sum(1 for _ in self._iter_batches())
        return self._num_batches


def batch_token_stats(batch: dict[str, Any]) -> dict[str, int]:
    cu_q = batch["cu_seqlens_q"]
    cu_k = batch["cu_seqlens_k"]
    return {
        "samples": int(cu_q.numel() - 1),
        "text_tokens": int(cu_q[-1].item()),
        "chunk_tokens": int(cu_k[-1].item()),
        "max_text_len": int(batch["max_seqlen_q"]),
        "max_chunk_len": int(batch["max_seqlen_k"]),
    }


def batch_label_tokens(batch: dict[str, Any]) -> int:
    return int((batch["labels"] != -100).sum().item())


def effective_total_steps(datasets: list[tuple[str, Dataset]], args: argparse.Namespace) -> int:
    micro_batches = sum(len(PackedBatchSampler(dataset, args, epoch=0)) for _, dataset in datasets)
    if micro_batches <= 0:
        raise ValueError("No full micro-batches are available. Reduce --micro-batch-size or check the datasets.")
    total_steps = micro_batches // args.grad_accum
    if total_steps <= 0:
        raise ValueError("No full optimizer steps are available. Reduce --grad-accum or --micro-batch-size.")
    return total_steps


def lr_for_step(step: int, args: argparse.Namespace) -> float:
    if step <= args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.lr_schedule_steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.lr * (args.min_lr_mult + (1.0 - args.min_lr_mult) * cosine)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def clip_grad_norm_tp(parameters: Iterable[torch.nn.Parameter], max_norm: float, group: dist.ProcessGroup) -> torch.Tensor:
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return torch.tensor(0.0, device=torch.cuda.current_device())

    total_sq = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
    for param in params:
        grad = param.grad.detach().float()
        total_sq += grad.pow(2).sum()
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=group)
    total_norm = total_sq.sqrt()

    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in params:
                param.grad.detach().mul_(clip_coef.to(param.grad.device, dtype=param.grad.dtype))
    return total_norm


def build_model(args: argparse.Namespace, config: CombLlamaConfig, tp_group: dist.ProcessGroup) -> CombLlamaForConditionalGeneration:
    model = CombLlamaForConditionalGeneration(config, from_scratch=args.from_scratch)
    apply_tensor_parallelism(model, tp_group)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    return model.to(device=torch.cuda.current_device(), dtype=dtype)


def dataset_metadata(datasets: list[tuple[str, Dataset]]) -> list[dict[str, Any]]:
    return [{"name": name, "length": len(dataset)} for name, dataset in datasets]


def checkpoint_payload(
    model,
    optimizer,
    step: int,
    config: CombLlamaConfig,
    args: argparse.Namespace,
    data_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "args": vars(args),
        "data_state": data_state,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def prune_old_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return

    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("step_*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            continue
        checkpoints.append((step, path))

    checkpoints.sort(key=lambda item: item[0])
    stale = checkpoints[:-keep_last_n]
    for _, path in stale:
        shutil.rmtree(path, ignore_errors=True)
        print0(f"removed old checkpoint: {path}")


def save_checkpoint(
    model,
    optimizer,
    step: int,
    config: CombLlamaConfig,
    args: argparse.Namespace,
    data_state: dict[str, Any],
) -> Path:
    out = Path(args.output_dir) / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    path = out / f"rank_{rank:05d}.pt"
    torch.save(checkpoint_payload(model, optimizer, step, config, args, data_state), path)
    if is_rank0():
        metadata = {"step": step, "world_size": dist.get_world_size(), "format": "tp-sharded", "data_state": data_state}
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print0(f"saved checkpoint: {out}")
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if is_rank0():
        prune_old_checkpoints(Path(args.output_dir), args.keep_last_n)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return out


def save_full_checkpoint(
    model,
    step: int,
    config: CombLlamaConfig,
    args: argparse.Namespace,
    data_state: dict[str, Any],
) -> None:
    local_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    gathered = [None for _ in range(dist.get_world_size())] if is_rank0() else None
    dist.gather_object(local_state, gathered, dst=0)
    if is_rank0():
        full_state = merge_state_dict_from_tp(gathered)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"step_{step:06d}_full.pt"
        torch.save(
            {"model": full_state, "step": step, "config": config.to_dict(), "args": vars(args), "data_state": data_state},
            path,
        )
        print0(f"saved full checkpoint: {path}")
    dist.barrier(device_ids=[torch.cuda.current_device()])


def validate_resume_datasets(saved: list[dict[str, Any]] | None, datasets: list[tuple[str, Dataset]]) -> None:
    if saved is None:
        return
    current = dataset_metadata(datasets)
    if saved != current:
        raise ValueError(f"Resume checkpoint datasets do not match current datasets: saved={saved}, current={current}")


def restore_rng_state(ckpt: dict[str, Any]) -> None:
    rng = ckpt.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng:
        torch.cuda.set_rng_state(rng["cuda"], device=torch.cuda.current_device())
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])


def load_resume(model, optimizer, resume: str, datasets: list[tuple[str, Dataset]]) -> tuple[int, dict[str, Any] | None]:
    path = Path(resume)
    if path.is_dir():
        path = path / f"rank_{dist.get_rank():05d}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    restore_rng_state(ckpt)
    data_state = ckpt.get("data_state")
    if data_state is not None:
        validate_resume_datasets(data_state.get("datasets"), datasets)
    print0(f"resumed from {path} at step {ckpt.get('step', 0)}")
    return int(ckpt.get("step", 0)), data_state


class OffsetBatchSampler(Sampler[list[int]]):
    """Skip already-consumed packed micro-batches without materializing dataset rows."""

    def __init__(self, batch_sampler: Sampler[list[int]], start_offset: int) -> None:
        self.batch_sampler = batch_sampler
        self.start_offset = start_offset

    def __iter__(self):
        for index, batch in enumerate(self.batch_sampler):
            if index >= self.start_offset:
                yield batch

    def __len__(self) -> int:
        return max(0, len(self.batch_sampler) - self.start_offset)


def make_loader(dataset: Dataset, args: argparse.Namespace, epoch: int, start_offset: int = 0) -> DataLoader:
    batch_sampler = PackedBatchSampler(dataset, args, epoch)
    if start_offset < 0 or start_offset > len(batch_sampler):
        raise ValueError(f"Invalid micro_batch_offset={start_offset}; sampler has {len(batch_sampler)} batches.")
    if is_rank0():
        print(
            f"packed batches={len(batch_sampler)} filtered={batch_sampler.filtered_count} "
            f"max_samples={args.micro_batch_size} max_text_tokens={args.max_text_tokens_per_batch} "
            f"max_chunk_tokens={args.max_chunk_tokens_per_batch} start_offset={start_offset}",
            flush=True,
        )
    if start_offset:
        batch_sampler = OffsetBatchSampler(batch_sampler, start_offset)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_fn, label_key=args.label_key),
    )


def evaluate_validation_loss(
    model,
    validation_datasets: list[tuple[str, Dataset]],
    args: argparse.Namespace,
    device: torch.device,
    step: int,
    validation_log_path: Path,
) -> None:
    if not validation_datasets or args.eval_interval <= 0:
        return

    was_training = model.training
    model.eval()
    overall_loss_sum = 0.0
    overall_label_tokens = 0
    rows = []

    with torch.no_grad():
        for dataset_name, dataset in validation_datasets:
            loader = make_loader(dataset, args, epoch=0)
            loss_sum = 0.0
            label_tokens_sum = 0
            micro_batches = 0
            stats_sum = {
                "samples": 0,
                "text_tokens": 0,
                "chunk_tokens": 0,
                "max_text_len": 0,
                "max_chunk_len": 0,
            }

            for batch in loader:
                if args.eval_max_batches > 0 and micro_batches >= args.eval_max_batches:
                    break
                stats = batch_token_stats(batch)
                label_tokens = batch_label_tokens(batch)
                batch = move_batch_to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    loss = model(**batch).loss

                loss_value = float(loss.detach())
                loss_sum += loss_value * label_tokens
                label_tokens_sum += label_tokens
                micro_batches += 1
                stats_sum["samples"] += stats["samples"]
                stats_sum["text_tokens"] += stats["text_tokens"]
                stats_sum["chunk_tokens"] += stats["chunk_tokens"]
                stats_sum["max_text_len"] = max(stats_sum["max_text_len"], stats["max_text_len"])
                stats_sum["max_chunk_len"] = max(stats_sum["max_chunk_len"], stats["max_chunk_len"])

            if label_tokens_sum == 0:
                raise RuntimeError(f"No validation label tokens found for dataset {dataset_name}.")
            validation_loss = loss_sum / label_tokens_sum
            overall_loss_sum += loss_sum
            overall_label_tokens += label_tokens_sum
            rows.append(
                [
                    step,
                    dataset_name,
                    validation_loss,
                    micro_batches,
                    stats_sum["samples"],
                    stats_sum["text_tokens"],
                    stats_sum["chunk_tokens"],
                    label_tokens_sum,
                    stats_sum["max_text_len"],
                    stats_sum["max_chunk_len"],
                ]
            )
            print0(
                f"eval step={step} dataset={dataset_name} loss={validation_loss:.6f} "
                f"micro_batches={micro_batches} samples={stats_sum['samples']} "
                f"text_tokens={stats_sum['text_tokens']} chunk_tokens={stats_sum['chunk_tokens']}"
            )

    if overall_label_tokens:
        rows.append([step, "__mean__", overall_loss_sum / overall_label_tokens, "", "", "", "", overall_label_tokens, "", ""])

    if is_rank0():
        validation_log_path.parent.mkdir(parents=True, exist_ok=True)
        with validation_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    if was_training:
        model.train()
    dist.barrier(device_ids=[torch.cuda.current_device()])


def train_loop(
    model,
    optimizer,
    datasets: list[tuple[str, Dataset]],
    validation_datasets: list[tuple[str, Dataset]],
    start_step: int,
    resume_data_state: dict[str, Any] | None,
    config: CombLlamaConfig,
    args: argparse.Namespace,
) -> None:
    if not datasets:
        raise ValueError("At least one training dataset is required.")
    total_steps = args.total_steps if args.total_steps is not None else effective_total_steps(datasets, args)
    if start_step >= total_steps:
        print0(f"start_step={start_step} already reached total_steps={total_steps}; nothing to train.")
        return
    print0(f"total optimizer steps: {total_steps}")
    print0(f"lr schedule steps: {args.lr_schedule_steps}")

    out = Path(args.output_dir)
    diag_path = out / "training_log.csv"
    comb_loss_path = out / "training_loss.csv"
    validation_log_path = out / "validation_log.csv"
    if is_rank0() and start_step == 0:
        out.mkdir(parents=True, exist_ok=True)
        with diag_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "dataset",
                    "loss",
                    "instant_loss",
                    "lr",
                    "grad_norm",
                    "instant_grad_norm",
                    "step_time_sec",
                    "total_time_sec",
                    "steps",
                    "micro_batches",
                    "samples",
                    "text_tokens",
                    "chunk_tokens",
                    "label_tokens",
                    "max_text_len",
                    "max_chunk_len",
                ]
            )
        comb_loss_path.write_text("", encoding="utf-8")
        with validation_log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "dataset",
                    "loss",
                    "micro_batches",
                    "samples",
                    "text_tokens",
                    "chunk_tokens",
                    "label_tokens",
                    "max_text_len",
                    "max_chunk_len",
                ]
            )

    step = start_step
    if resume_data_state is None:
        epoch = 0
        dataset_index = 0
        micro_batch_offset = 0
    else:
        epoch = int(resume_data_state.get("epoch", 0))
        dataset_index = int(resume_data_state.get("dataset_index", 0))
        micro_batch_offset = int(resume_data_state.get("micro_batch_offset", 0))
        if dataset_index < 0 or dataset_index >= len(datasets):
            raise ValueError(f"Invalid resumed dataset_index={dataset_index}; have {len(datasets)} datasets.")
    device = torch.device("cuda", torch.cuda.current_device())
    model.train()
    dataset_name, dataset = datasets[dataset_index]
    print0(
        f"training dataset: {dataset_name} size={len(dataset)} epoch={epoch} "
        f"micro_batch_offset={micro_batch_offset}"
    )
    loader = make_loader(dataset, args, epoch, start_offset=micro_batch_offset)
    iterator = iter(loader)

    def current_data_state() -> dict[str, Any]:
        return {
            "dataset_index": dataset_index,
            "dataset_name": dataset_name,
            "epoch": epoch,
            "micro_batch_offset": micro_batch_offset,
            "datasets": dataset_metadata(datasets),
        }

    def advance_dataset() -> str:
        nonlocal dataset_index, dataset_name, dataset, epoch, micro_batch_offset, loader, iterator
        completed_dataset_name = dataset_name
        dataset_index += 1
        if dataset_index >= len(datasets):
            dataset_index = 0
            epoch += 1
        dataset_name, dataset = datasets[dataset_index]
        micro_batch_offset = 0
        print0(
            f"training dataset: {dataset_name} size={len(dataset)} epoch={epoch} "
            f"micro_batch_offset={micro_batch_offset}"
        )
        loader = make_loader(dataset, args, epoch, start_offset=micro_batch_offset)
        iterator = iter(loader)
        return completed_dataset_name

    interval_loss_weighted = 0.0
    interval_label_tokens = 0
    interval_steps = 0
    interval_micro_batches = 0
    interval_grad_norm_sum = 0.0
    interval_time_sum = 0.0
    interval_stats = {
        "samples": 0,
        "text_tokens": 0,
        "chunk_tokens": 0,
        "max_text_len": 0,
        "max_chunk_len": 0,
    }
    last_instant_loss = 0.0
    last_instant_grad_norm = 0.0
    last_log_dataset_name = dataset_name

    def reset_interval() -> None:
        nonlocal interval_loss_weighted, interval_label_tokens, interval_steps, interval_micro_batches
        nonlocal interval_grad_norm_sum, interval_time_sum, interval_stats
        interval_loss_weighted = 0.0
        interval_label_tokens = 0
        interval_steps = 0
        interval_micro_batches = 0
        interval_grad_norm_sum = 0.0
        interval_time_sum = 0.0
        interval_stats = {
            "samples": 0,
            "text_tokens": 0,
            "chunk_tokens": 0,
            "max_text_len": 0,
            "max_chunk_len": 0,
        }

    def write_training_log() -> None:
        if interval_steps == 0 or not is_rank0():
            return
        avg_loss = interval_loss_weighted / interval_label_tokens if interval_label_tokens else last_instant_loss
        avg_grad_norm = interval_grad_norm_sum / interval_steps
        avg_step_time = interval_time_sum / interval_steps
        print(
            f"step={step} dataset={last_log_dataset_name} loss={avg_loss:.6f} "
            f"instant_loss={last_instant_loss:.6f} lr={lr:.6g} grad_norm={avg_grad_norm:.6f} "
            f"instant_grad_norm={last_instant_grad_norm:.6f} step_time={avg_step_time:.3f}s "
            f"steps={interval_steps} micro_batches={interval_micro_batches} "
            f"samples={interval_stats['samples']} text_tokens={interval_stats['text_tokens']} "
            f"chunk_tokens={interval_stats['chunk_tokens']} label_tokens={interval_label_tokens}",
            flush=True,
        )
        with diag_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    step,
                    last_log_dataset_name,
                    avg_loss,
                    last_instant_loss,
                    lr,
                    avg_grad_norm,
                    last_instant_grad_norm,
                    avg_step_time,
                    interval_time_sum,
                    interval_steps,
                    interval_micro_batches,
                    interval_stats["samples"],
                    interval_stats["text_tokens"],
                    interval_stats["chunk_tokens"],
                    interval_label_tokens,
                    interval_stats["max_text_len"],
                    interval_stats["max_chunk_len"],
                ]
            )
        with comb_loss_path.open("a", newline="") as f:
            csv.writer(f).writerow([last_log_dataset_name, step - 1, avg_loss])
        reset_interval()

    while step < total_steps:
        micro_batches = []
        log_dataset_name = dataset_name
        completed_dataset_name = None
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                if micro_batches:
                    completed_dataset_name = dataset_name
                    break

                write_training_log()
                just_completed_dataset_name = advance_dataset()
                print0(
                    f"completed dataset: {just_completed_dataset_name}; "
                    f"saving boundary checkpoint at step={step}"
                )
                save_checkpoint(model, optimizer, step, config, args, current_data_state())
                batch = next(iterator)
            micro_batches.append(batch)
            micro_batch_offset += 1
            log_dataset_name = dataset_name

        optimizer.zero_grad(set_to_none=True)
        lr = lr_for_step(step + 1, args)
        set_lr(optimizer, lr)
        step_start = time.perf_counter()

        loss_sum = 0.0
        loss_weighted_sum = 0.0
        label_tokens_sum = 0
        processed_micro_batches = 0
        log_stats = {
            "samples": 0,
            "text_tokens": 0,
            "chunk_tokens": 0,
            "max_text_len": 0,
            "max_chunk_len": 0,
        }
        for batch in micro_batches:
            stats = batch_token_stats(batch)
            log_stats["samples"] += stats["samples"]
            log_stats["text_tokens"] += stats["text_tokens"]
            log_stats["chunk_tokens"] += stats["chunk_tokens"]
            log_stats["max_text_len"] = max(log_stats["max_text_len"], stats["max_text_len"])
            log_stats["max_chunk_len"] = max(log_stats["max_chunk_len"], stats["max_chunk_len"])

            label_tokens = batch_label_tokens(batch)
            batch = move_batch_to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw_loss = model(**batch).loss
            (raw_loss / len(micro_batches)).backward()
            raw_loss_value = float(raw_loss.detach())
            loss_sum += raw_loss_value
            loss_weighted_sum += raw_loss_value * label_tokens
            label_tokens_sum += label_tokens
            processed_micro_batches += 1

        grad_norm = clip_grad_norm_tp(model.parameters(), args.grad_clip, dist.group.WORLD)
        optimizer.step()
        torch.cuda.synchronize()
        step += 1

        if processed_micro_batches == 0:
            raise RuntimeError("No micro-batches were processed before optimizer step.")
        log_loss_value = loss_weighted_sum / label_tokens_sum if label_tokens_sum else loss_sum / processed_micro_batches
        grad_norm_value = float(grad_norm.detach().float().item())
        step_time = time.perf_counter() - step_start

        interval_loss_weighted += loss_weighted_sum
        interval_label_tokens += label_tokens_sum
        interval_steps += 1
        interval_micro_batches += processed_micro_batches
        interval_grad_norm_sum += grad_norm_value
        interval_time_sum += step_time
        interval_stats["samples"] += log_stats["samples"]
        interval_stats["text_tokens"] += log_stats["text_tokens"]
        interval_stats["chunk_tokens"] += log_stats["chunk_tokens"]
        interval_stats["max_text_len"] = max(interval_stats["max_text_len"], log_stats["max_text_len"])
        interval_stats["max_chunk_len"] = max(interval_stats["max_chunk_len"], log_stats["max_chunk_len"])
        last_instant_loss = log_loss_value
        last_instant_grad_norm = grad_norm_value
        last_log_dataset_name = log_dataset_name
        if step == 1 or step % args.log_interval == 0:
            write_training_log()

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, config, args, current_data_state())

        if completed_dataset_name is not None:
            write_training_log()
            advance_dataset()
            print0(f"completed dataset: {completed_dataset_name}; saving boundary checkpoint at step={step}")
            save_checkpoint(model, optimizer, step, config, args, current_data_state())

        if validation_datasets and args.eval_interval > 0 and step % args.eval_interval == 0:
            evaluate_validation_loss(model, validation_datasets, args, device, step, validation_log_path)

    if args.save_final:
        write_training_log()
        save_checkpoint(model, optimizer, step, config, args, current_data_state())
    if args.save_full_final:
        save_full_checkpoint(model, step, config, args, current_data_state())


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    seed_all(args.seed)

    tp_group = dist.group.WORLD
    config = load_config(args)
    if config.get_text_config().vocab_size % dist.get_world_size(tp_group) != 0:
        raise ValueError("vocab_size must divide TP world size for vocab-parallel lm_head.")

    datasets = load_datasets(args)
    train_datasets, validation_datasets = split_train_validation(datasets, args)
    print0("datasets:")
    for dataset_name, dataset in train_datasets:
        validation_size = next((len(val_dataset) for val_name, val_dataset in validation_datasets if val_name == dataset_name), 0)
        print0(f"  {dataset_name}: train={len(dataset)} validation={validation_size}")
    model = build_model(args, config, tp_group)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if args.resume:
        start_step, resume_data_state = load_resume(model, optimizer, args.resume, train_datasets)
    else:
        start_step, resume_data_state = 0, None
    train_loop(model, optimizer, train_datasets, validation_datasets, start_step, resume_data_state, config, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
