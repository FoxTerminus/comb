"""Reproducible, resumable launcher for the released CombLlama training recipe.

The model, data order, bucketing, loss, and DeepSpeed configuration are kept
identical to ``train_llama.py``.  This wrapper only adds bounded smoke runs,
optimizer-step logging, and periodic resumable checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

import deepspeed
import pyarrow.compute as pc
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from comb.integration.hf.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
from data import DATASET_DICT, TRAIN_DATASETS
from data.base import CPU_NUM, collate_fn
from data.base import BUCKET_SIZE


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CONTEXT_EVAL_DATA = (
    "/data3/junhaohu/.cache/huggingface/datasets/"
    "SQuAD_meta-llama_Llama-3.1-8B-Instruct"
)

# Specialized entrypoints can extend the reproducibility snapshot without
# duplicating the generic training loop.  Keys are paths inside the snapshot;
# values may be absolute paths to the actual sources used by the wrapper.
ADDITIONAL_SOURCE_SNAPSHOT_FILES: dict[str, Path] = {}


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ds_llama_config.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--max-samples-per-dataset", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--export-final-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export a consolidated Hugging Face checkpoint after training.",
    )
    parser.add_argument("--resume-root")
    parser.add_argument("--resume-tag")
    parser.add_argument(
        "--resume-reset-data-position",
        action="store_true",
        help="Load model/optimizer state but begin the configured dataset stage at its first row.",
    )
    parser.add_argument(
        "--resume-reset-lr-scheduler",
        action="store_true",
        help="Load model/optimizer state without restoring the previous LR scheduler state.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Load a resumable checkpoint and export HF weights without training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Load a resumable checkpoint and run the context-dependency "
            "evaluation without performing an optimizer step."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-eval-examples", type=int, default=64)
    parser.add_argument("--context-eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--context-eval-suite-manifest",
        help=(
            "Optional JSON manifest of fixed context-dependency datasets. "
            "When omitted, retain the historical SQuAD-train evaluation."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help=(
            "DeepSpeed TP process-group size. The released entry path does not "
            "call tp_model_init, so values above one replicate rather than shard "
            "the model in DeepSpeed 0.17.2."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def rank0() -> bool:
    return torch.distributed.get_rank() == 0


def append_loss(path: Path, row: list[object]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["optimizer_step", "dataset", "bucket", "batch", "loss", "lr"])
        writer.writerow(row)


def audit_training_rows(wrapped) -> dict[str, int | str]:
    """Report official truncation/bucketing edge cases without changing data."""
    table = wrapped.data.data.table
    query_lengths = pc.list_value_length(table["input_ids"])
    context_lengths = table["token_count"]
    query_valid = pc.less_equal(query_lengths, wrapped.max_input_length)
    context_valid = pc.and_(
        pc.greater(context_lengths, BUCKET_SIZE[0]),
        pc.less_equal(context_lengths, BUCKET_SIZE[-1]),
    )
    return {
        "dataset_fingerprint": str(
            getattr(wrapped.data, "_fingerprint", "unavailable")
        ),
        "input_rows": len(wrapped.data),
        "rows_inside_context_buckets": int(
            pc.sum(pc.cast(context_valid, "int64")).as_py()
        ),
        "query_rows_fully_masked_by_truncation": int(
            pc.sum(pc.cast(pc.invert(query_valid), "int64")).as_py()
        ),
        "context_rows_outside_buckets": int(
            pc.sum(pc.cast(pc.invert(context_valid), "int64")).as_py()
        ),
    }


def slice_bucket_for_resume(dataset, batch_size: int, skip_batches: int):
    """Skip completed batches by Arrow row offset without changing order."""
    if skip_batches < 0:
        raise ValueError("skip_batches must be non-negative")
    completed_rows = skip_batches * batch_size
    if completed_rows > len(dataset):
        raise ValueError(
            f"Checkpoint resumes after row {completed_rows}, but bucket only "
            f"contains {len(dataset)} rows"
        )
    if completed_rows:
        dataset = dataset.select(range(completed_rows, len(dataset)))
    return dataset, skip_batches


def save_checkpoint(
    engine,
    output_dir: Path,
    optimizer_step: int,
    dataset_index: int,
    bucket_index: int,
    next_batch_index: int,
    keep_last_checkpoints: int,
) -> None:
    tag = f"step_{optimizer_step:08d}"
    engine.save_checkpoint(
        str(output_dir),
        tag=tag,
        client_state={
            "optimizer_step": optimizer_step,
            "dataset_index": dataset_index,
            "bucket_index": bucket_index,
            "next_batch_index": next_batch_index,
            "git_commit": git_commit(),
        },
    )
    torch.distributed.barrier()
    if rank0():
        (output_dir / "latest_repro.json").write_text(
            json.dumps({"tag": tag, "optimizer_step": optimizer_step}, indent=2) + "\n"
        )
        print(f"saved checkpoint {tag}", flush=True)
        if keep_last_checkpoints > 0:
            checkpoint_pattern = re.compile(r"^step_(\d{8})$")
            root = output_dir.resolve()
            checkpoints = sorted(
                (
                    path
                    for path in root.iterdir()
                    if path.is_dir() and checkpoint_pattern.fullmatch(path.name)
                ),
                key=lambda path: int(checkpoint_pattern.fullmatch(path.name).group(1)),
            )
            for old_checkpoint in checkpoints[:-keep_last_checkpoints]:
                resolved = old_checkpoint.resolve()
                if resolved.parent != root or not checkpoint_pattern.fullmatch(resolved.name):
                    raise RuntimeError(f"Refusing unsafe checkpoint cleanup: {resolved}")
                shutil.rmtree(resolved)
                print(f"removed expired checkpoint {resolved.name}", flush=True)
    torch.distributed.barrier()


def export_hf_checkpoint(engine, output_dir: Path, optimizer_step: int) -> None:
    """Consolidate the DeepSpeed state and export a loadable HF checkpoint."""
    export_dir = output_dir / f"hf_step_{optimizer_step:08d}"
    engine.eval()
    # Use DeepSpeed's consolidation path so this remains correct for ZeRO and
    # for any future launcher that performs actual model-parallel replacement.
    state_dict = engine._consolidated_16bit_state_dict()
    if rank0():
        engine.module.save_pretrained(
            export_dir,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        print(f"exported Hugging Face checkpoint {export_dir}", flush=True)
    del state_dict
    torch.distributed.barrier()
    engine.train()


def evaluate_context_dependency(
    engine,
    output_dir: Path,
    optimizer_step: int,
    examples: int,
    batch_size: int,
    suite_manifest: str | None = None,
) -> None:
    """Evaluate the exact loaded state without mutating model parameters."""
    if not examples:
        return
    from benchmarks.context_dependency_eval import evaluate_model

    engine.eval()
    if suite_manifest is None:
        entries = [
            {
                "name": None,
                "path": CONTEXT_EVAL_DATA,
                "target_column": None,
                "examples": examples,
                "batch_size": batch_size,
            }
        ]
        suite_metadata = None
    else:
        suite_path = Path(suite_manifest).resolve()
        suite_metadata = json.loads(suite_path.read_text())
        if suite_metadata.get("version") != 1:
            raise ValueError("context-eval suite manifest version must be 1")
        entries = suite_metadata.get("datasets")
        if not isinstance(entries, list) or not entries:
            raise ValueError("context-eval suite manifest requires non-empty datasets")
        if rank0():
            copied_manifest = output_dir / "context_eval_suite_manifest.json"
            manifest_text = suite_path.read_text()
            if copied_manifest.exists() and copied_manifest.read_text() != manifest_text:
                raise RuntimeError(
                    f"refusing to overwrite different suite manifest: {copied_manifest}"
                )
            copied_manifest.write_text(manifest_text)

    suite_summary = {}
    for entry in entries:
        name = entry.get("name")
        if name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe context-eval dataset name: {name!r}")
        data_path = str(Path(entry["path"]).resolve())
        target_column = entry.get("target_column")
        entry_examples = int(entry.get("examples", examples))
        entry_batch_size = int(entry.get("batch_size", batch_size))
        if entry_examples < 2 or entry_batch_size < 1:
            raise ValueError(f"invalid context-eval sizing for {name!r}")
        context_dataset = load_from_disk(data_path)
        evaluation_kwargs = {}
        if target_column is not None:
            evaluation_kwargs["target_column"] = target_column
        context_result = evaluate_model(
            engine.module,
            context_dataset,
            entry_examples,
            entry_batch_size,
            **evaluation_kwargs,
        )
        context_result["optimizer_step"] = optimizer_step
        if name is not None:
            context_result.update(
                {
                    "dataset_name": name,
                    "data_path": data_path,
                    "source_dataset_fingerprint": str(
                        getattr(context_dataset, "_fingerprint", "unavailable")
                    ),
                    "target_column": target_column or "default_teacher_column",
                }
            )
        if rank0():
            if name is None:
                filename = f"context_dependency_step_{optimizer_step:08d}.json"
            else:
                filename = (
                    f"context_dependency_{name}_step_{optimizer_step:08d}.json"
                )
            context_path = output_dir / filename
            context_path.write_text(json.dumps(context_result, indent=2) + "\n")
            print(json.dumps(context_result), flush=True)
            if name is not None:
                summary_keys = (
                    "examples",
                    "target_tokens",
                    "correct_context_nll",
                    "shuffled_context_nll",
                    "context_nll_gap",
                    "positive_batch_fraction",
                    "positive_example_fraction",
                    "distinct_context_nll",
                    "distinct_context_nll_gap",
                    "distinct_positive_batch_fraction",
                    "distinct_positive_example_fraction",
                    "no_context_nll",
                    "no_context_nll_gap",
                    "no_context_positive_example_fraction",
                    "mean_abs_tanh_gate",
                )
                suite_summary[name] = {
                    key: context_result[key]
                    for key in summary_keys
                    if key in context_result
                }
    if rank0() and suite_metadata is not None:
        suite_result = {
            "optimizer_step": optimizer_step,
            "suite_manifest": str(Path(suite_manifest).resolve()),
            "suite_manifest_sha256": sha256_file(Path(suite_manifest).resolve()),
            "suite_name": suite_metadata.get("name"),
            "datasets": suite_summary,
        }
        suite_path = output_dir / f"context_dependency_suite_step_{optimizer_step:08d}.json"
        suite_path.write_text(json.dumps(suite_result, indent=2) + "\n")
    torch.distributed.barrier()


def main() -> None:
    args = parse_args()
    if args.export_only and not args.resume_root:
        raise ValueError("--export-only requires --resume-root")
    if args.eval_only and not args.resume_root:
        raise ValueError("--eval-only requires --resume-root")
    if args.eval_only and args.export_only:
        raise ValueError("--eval-only and --export-only are mutually exclusive")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CombLlamaForConditionalGeneration(
        from_scratch=True,
        config=CombLlamaConfig(LlamaConfig.from_pretrained(MODEL_NAME)),
    )
    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=7200))
    config = json.loads(Path(args.config).read_text())
    config["ckpt_folder"] = None
    if config.get("tensorboard", {}).get("enabled"):
        config["tensorboard"].setdefault("output_path", str(output_dir / "tensorboard"))
    config["tensor_parallel"]["autotp_size"] = args.tensor_parallel_size
    config["tensor_parallel"]["tensor_parallel"]["tp_size"] = (
        args.tensor_parallel_size
    )
    world_size = torch.distributed.get_world_size()
    if world_size % args.tensor_parallel_size:
        raise ValueError(
            f"world size {world_size} must be divisible by TP size "
            f"{args.tensor_parallel_size}"
        )
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    engine.train()

    optimizer_step = int(engine.global_steps)
    start_dataset = 0
    start_bucket = 0
    start_batch = 0
    if args.resume_root:
        _, state = engine.load_checkpoint(
            args.resume_root,
            tag=args.resume_tag,
            load_lr_scheduler_states=not args.resume_reset_lr_scheduler,
        )
        if state is None:
            raise RuntimeError("DeepSpeed did not return checkpoint client state")
        optimizer_step = int(state["optimizer_step"])
        if args.resume_reset_data_position:
            start_dataset = 0
            start_bucket = 0
            start_batch = 0
        else:
            start_dataset = int(state["dataset_index"])
            start_bucket = int(state["bucket_index"])
            start_batch = int(state["next_batch_index"])

    if rank0():
        trainable = sum(p.numel() for p in engine.module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in engine.module.parameters())
        source_files = {
            "training/train_llama_repro.py": Path("train_llama_repro.py"),
            f"training/{Path(args.config).name}": Path(args.config),
            "comb/integration/hf/CombLlama.py": Path(
                "../comb/integration/hf/CombLlama.py"
            ),
            "data/base.py": Path("../data/base.py"),
            "benchmarks/context_dependency_eval.py": Path(
                "../benchmarks/context_dependency_eval.py"
            ),
        }
        source_files.update(ADDITIONAL_SOURCE_SNAPSHOT_FILES)
        source_hashes = {
            str(path): sha256_file(path) for path in source_files.values()
        }
        snapshot_name = (
            f"source_snapshot_start_{optimizer_step:08d}_"
            f"{source_hashes['train_llama_repro.py'][:12]}"
        )
        snapshot_dir = output_dir / snapshot_name
        for relative_name, source_path in source_files.items():
            snapshot_path = snapshot_dir / relative_name
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            if snapshot_path.exists() and sha256_file(snapshot_path) != sha256_file(
                source_path
            ):
                raise RuntimeError(f"Refusing to overwrite source snapshot {snapshot_path}")
            shutil.copy2(source_path, snapshot_path)
        manifest = {
            "model": MODEL_NAME,
            "official_git_commit": git_commit(),
            "start_optimizer_step": optimizer_step,
            "resume_root": args.resume_root,
            "resume_tag": args.resume_tag,
            "resume_reset_data_position": args.resume_reset_data_position,
            "resume_reset_lr_scheduler": args.resume_reset_lr_scheduler,
            "source_snapshot": snapshot_name,
            "software": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torch_cuda": torch.version.cuda,
                "transformers": package_version("transformers"),
                "deepspeed": package_version("deepspeed"),
                "datasets": package_version("datasets"),
            },
            "hardware": {
                "world_size": world_size,
                "visible_gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            },
            "parallelism": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "data_parallel_size": engine.dp_world_size,
                "effective_global_batch_size": config["train_batch_size"],
                "deepspeed_gradient_accumulation_steps": config[
                    "gradient_accumulation_steps"
                ],
                "deepspeed_autotp_model_replaced": bool(
                    getattr(engine.module, "ds_autotp_parsed", False)
                ),
                "checkpoint_rank_semantics": (
                    "autotp_partition"
                    if getattr(engine.module, "ds_autotp_parsed", False)
                    else "complete_model"
                ),
            },
            "datasets": TRAIN_DATASETS,
            "deepspeed_config": config,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "seed": args.seed,
            "max_samples_per_dataset": args.max_samples_per_dataset,
            "source_sha256": source_hashes,
        }
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        latest_manifest = output_dir / "run_manifest.json"
        if latest_manifest.exists():
            previous = json.loads(latest_manifest.read_text())
            previous_start = int(previous.get("start_optimizer_step", 0))
            previous_path = (
                output_dir / f"run_manifest_start_{previous_start:08d}.json"
            )
            if not previous_path.exists():
                previous_path.write_text(latest_manifest.read_text())
        (output_dir / f"run_manifest_start_{optimizer_step:08d}.json").write_text(
            manifest_text
        )
        latest_manifest.write_text(manifest_text)
        print(json.dumps(manifest), flush=True)

    if args.export_only:
        export_hf_checkpoint(engine, output_dir, optimizer_step)
        torch.distributed.destroy_process_group()
        return
    if args.eval_only:
        evaluate_context_dependency(
            engine,
            output_dir,
            optimizer_step,
            args.context_eval_examples,
            args.context_eval_batch_size,
            args.context_eval_suite_manifest,
        )
        torch.distributed.destroy_process_group()
        return

    loss_path = output_dir / "training_loss.csv"
    should_stop = False
    final_position = (start_dataset, start_bucket, start_batch)
    for dataset_index, dataset_name in enumerate(TRAIN_DATASETS):
        if dataset_index < start_dataset:
            continue
        wrapped = DATASET_DICT[dataset_name](MODEL_NAME, split="train")
        if args.max_samples_per_dataset:
            wrapped.data = wrapped.data.select(
                range(min(args.max_samples_per_dataset, len(wrapped.data)))
            )
        if rank0():
            audit = audit_training_rows(wrapped)
            (output_dir / f"data_audit_{dataset_name}.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            print(json.dumps({"dataset": dataset_name, **audit}), flush=True)
        buckets = wrapped.bucketing(engine.local_rank, torch.distributed.get_world_size())
        for bucket_index, (batch_size, parquet_file) in enumerate(buckets):
            if dataset_index == start_dataset and bucket_index < start_bucket:
                continue
            skip_batches = (
                start_batch
                if dataset_index == start_dataset and bucket_index == start_bucket
                else 0
            )
            bucket_dataset, batch_index_offset = slice_bucket_for_resume(
                Dataset.from_parquet(parquet_file), batch_size, skip_batches
            )
            loader = DataLoader(
                bucket_dataset,
                collate_fn=collate_fn,
                batch_size=batch_size,
                shuffle=False,
                num_workers=(
                    args.num_workers
                    if args.num_workers is not None
                    else max(CPU_NUM // engine.world_size, 1)
                ),
                drop_last=False,
            )
            accumulation_groups = (
                config["train_batch_size"]
                // engine.dp_world_size
                // config["gradient_accumulation_steps"]
                // batch_size
            )
            if accumulation_groups < 1:
                raise ValueError(f"Invalid official accumulation for bucket batch size {batch_size}")
            pending_batches = 0
            last_loss = None
            last_batch_index = -1
            for relative_batch_index, batch in enumerate(loader):
                batch_index = relative_batch_index + batch_index_offset
                if not torch.any(batch["shift_labels"] != -100):
                    raise RuntimeError(
                        f"All labels are ignored in {dataset_name} bucket={bucket_index} "
                        f"batch={batch_index}"
                    )
                batch = {key: value.to(engine.device) for key, value in batch.items()}
                output = engine(**batch)
                loss = output.loss
                engine.backward(loss)
                pending_batches += 1
                last_loss = loss
                last_batch_index = batch_index
                final_position = (dataset_index, bucket_index, batch_index + 1)
                if (batch_index + 1) % accumulation_groups != 0:
                    continue

                before = int(engine.global_steps)
                engine.step()
                pending_batches = 0
                after = int(engine.global_steps)
                if after == before:
                    continue
                optimizer_step = after
                if rank0() and optimizer_step % args.log_interval == 0:
                    lr = float(engine.get_lr()[0])
                    append_loss(
                        loss_path,
                        [
                            optimizer_step,
                            dataset_name,
                            bucket_index,
                            batch_index,
                            float(loss.detach()),
                            lr,
                        ],
                    )
                    print(
                        f"step={optimizer_step} dataset={dataset_name} bucket={bucket_index} "
                        f"batch={batch_index} loss={float(loss.detach()):.6f} lr={lr:.8g}",
                        flush=True,
                    )
                if args.save_interval and optimizer_step % args.save_interval == 0:
                    save_checkpoint(
                        engine,
                        output_dir,
                        optimizer_step,
                        dataset_index,
                        bucket_index,
                        batch_index + 1,
                        args.keep_last_checkpoints,
                    )
                if args.max_optimizer_steps and optimizer_step >= args.max_optimizer_steps:
                    should_stop = True
                    break
            # Preserve the official script's unconditional bucket-end step.
            # DeepSpeed counts calls to engine.step() as accumulation micro-steps,
            # so omitting this call can shift later optimizer boundaries even when
            # there is no partial manual accumulation group.
            if not should_stop:
                before = int(engine.global_steps)
                engine.step()
                after = int(engine.global_steps)
                if after > before:
                    optimizer_step = after
                    if (
                        rank0()
                        and last_loss is not None
                        and optimizer_step % args.log_interval == 0
                    ):
                        lr = float(engine.get_lr()[0])
                        append_loss(
                            loss_path,
                            [
                                optimizer_step,
                                dataset_name,
                                bucket_index,
                                last_batch_index,
                                float(last_loss.detach()),
                                lr,
                            ],
                        )
                        print(
                            f"step={optimizer_step} dataset={dataset_name} "
                            f"bucket={bucket_index} batch={last_batch_index} "
                            f"loss={float(last_loss.detach()):.6f} lr={lr:.8g} "
                            f"official_bucket_end_step=true "
                            f"partial_bucket={bool(pending_batches)}",
                            flush=True,
                        )
                    elif rank0() and optimizer_step % args.log_interval == 0:
                        print(
                            f"step={optimizer_step} dataset={dataset_name} "
                            f"bucket={bucket_index} official_bucket_end_step=true "
                            "loss_unavailable_after_resume=true",
                            flush=True,
                        )
                    if args.save_interval and optimizer_step % args.save_interval == 0:
                        save_checkpoint(
                            engine,
                            output_dir,
                            optimizer_step,
                            dataset_index,
                            bucket_index + 1,
                            0,
                            args.keep_last_checkpoints,
                        )
                    if args.max_optimizer_steps and optimizer_step >= args.max_optimizer_steps:
                        should_stop = True
            start_batch = 0
            if should_stop:
                break
        start_bucket = 0
        if should_stop:
            break

    if args.save_final and (
        optimizer_step == 0
        or not (args.save_interval and optimizer_step % args.save_interval == 0)
    ):
        save_checkpoint(
            engine,
            output_dir,
            optimizer_step,
            *final_position,
            args.keep_last_checkpoints,
        )
    evaluate_context_dependency(
        engine,
        output_dir,
        optimizer_step,
        args.context_eval_examples,
        args.context_eval_batch_size,
        args.context_eval_suite_manifest,
    )
    if args.export_final_hf:
        export_hf_checkpoint(engine, output_dir, optimizer_step)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
