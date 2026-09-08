#!/usr/bin/env python3
"""Capacity-test a native Comb AutoTP checkpoint on a real long-context row.

This is intentionally separate from the reproduction trainer.  It loads the
same model, DeepSpeed configuration, optimizer state, and TP adapter, runs one
forward/backward/AdamW update in memory, records rank-local CUDA peaks, and
exits without saving the mutated state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import deepspeed
import torch
from datasets import Dataset
from transformers import LlamaConfig

from comb.integration.hf.CombLlama import (
    CombLlamaConfig,
    CombLlamaForConditionalGeneration,
)
from data.base import collate_fn
from training.artifact_io import write_text_atomic
from training.train_llama_true_tp_repro import install_true_tp_hooks


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--resume-tag", required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "total_bytes": int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory),
    }


def gather_rank_reports(report: dict[str, object]) -> list[dict[str, object]]:
    gathered: list[dict[str, object] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, report)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather every TP rank capacity report")
    return [item for item in gathered if item is not None]


def optimizer_state_steps(engine) -> list[int]:
    """Read the restored basic AdamW step counters without copying tensors."""
    optimizer = getattr(engine.optimizer, "optimizer", None)
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("restored basic optimizer has no state")
    steps = []
    for value in state.values():
        if not isinstance(value, dict) or "step" not in value:
            raise RuntimeError("restored optimizer state has no Adam step")
        step = value["step"]
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise RuntimeError("Adam step tensor must be scalar")
            step = step.item()
        steps.append(int(step))
    return sorted(steps)


def capacity_checks(
    ranks: list[dict[str, object]],
    *,
    context_tokens: int,
    context_nonpadding_tokens: int,
    query_tokens: int,
) -> dict[str, object]:
    if not ranks:
        raise ValueError("capacity checks require at least one rank report")
    losses = [float(item["loss"]) for item in ranks]
    finite_loss_all_ranks = all(math.isfinite(value) for value in losses)
    loss_max_abs_delta_across_ranks = max(losses) - min(losses)
    shapes_consistent_across_ranks = all(
        int(item["context_tokens"]) == context_tokens
        and int(item["context_nonpadding_tokens"]) == context_nonpadding_tokens
        and int(item["query_tokens"]) == query_tokens
        for item in ranks
    )
    phases = (
        "baseline",
        "after_forward",
        "after_backward",
        "after_optimizer",
    )
    cuda_memory_accounting_valid = all(
        0 <= int(item[phase]["allocated_bytes"])
        <= int(item[phase]["peak_allocated_bytes"])
        <= int(item[phase]["peak_reserved_bytes"])
        <= int(item[phase]["total_bytes"])
        and int(item[phase]["allocated_bytes"])
        <= int(item[phase]["reserved_bytes"])
        <= int(item[phase]["peak_reserved_bytes"])
        for item in ranks
        for phase in phases
    )
    minimum_peak_reserved_headroom_bytes = min(
        int(item["after_optimizer"]["total_bytes"])
        - int(item["after_optimizer"]["peak_reserved_bytes"])
        for item in ranks
    )
    optimizer_update_observed_all_ranks = all(
        item.get("optimizer_state_step_advanced") is True
        and item.get("parameter_update_observed") is True
        and math.isfinite(float(item.get("parameter_update_max_abs_delta", float("nan"))))
        and float(item.get("parameter_update_max_abs_delta", 0.0)) > 0.0
        and int(item.get("engine_global_steps_before", -1))
        == int(item.get("engine_global_steps_after", -2))
        for item in ranks
    )
    passed = (
        finite_loss_all_ranks
        and loss_max_abs_delta_across_ranks <= 1e-6
        and shapes_consistent_across_ranks
        and cuda_memory_accounting_valid
        and minimum_peak_reserved_headroom_bytes >= 0
        and optimizer_update_observed_all_ranks
    )
    return {
        "finite_loss_all_ranks": finite_loss_all_ranks,
        "loss_max_abs_delta_across_ranks": loss_max_abs_delta_across_ranks,
        "shapes_consistent_across_ranks": shapes_consistent_across_ranks,
        "cuda_memory_accounting_valid": cuda_memory_accounting_valid,
        "minimum_peak_reserved_headroom_bytes": minimum_peak_reserved_headroom_bytes,
        "optimizer_update_observed_all_ranks": optimizer_update_observed_all_ranks,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if args.tensor_parallel_size < 2:
        raise ValueError("capacity preflight requires true TP size >= 2")
    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # The true-TP adapter reads these ordinary trainer flags while wrapping
    # deepspeed.initialize().  Preserve the real CLI for the report first.
    invocation = list(sys.argv)
    sys.argv.extend(
        [
            "--resume-root",
            str(args.resume_root.resolve()),
            "--resume-tag",
            args.resume_tag,
            "--output-dir",
            str(args.output_dir.resolve()),
        ]
    )
    install_true_tp_hooks()
    torch.manual_seed(args.seed)
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != args.tensor_parallel_size:
        raise ValueError(
            f"launcher world size {world_size} != TP size {args.tensor_parallel_size}"
        )

    if rank == 0:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_tp2_checkpoint.py")),
                str(args.resume_root.resolve()),
                "--tag",
                args.resume_tag,
                "--model-parallel-size",
                str(args.tensor_parallel_size),
                "--allow-nonlatest",
            ],
            check=True,
        )
    torch.distributed.barrier()

    model = CombLlamaForConditionalGeneration(
        from_scratch=True,
        config=CombLlamaConfig(LlamaConfig.from_pretrained(MODEL_NAME)),
    )
    config = json.loads(args.config.read_text())
    config["ckpt_folder"] = None
    config["tensor_parallel"]["autotp_size"] = args.tensor_parallel_size
    config["tensor_parallel"]["tensor_parallel"]["tp_size"] = args.tensor_parallel_size
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    _, client_state = engine.load_checkpoint(
        str(args.resume_root.resolve()), tag=args.resume_tag
    )
    if client_state is None:
        raise RuntimeError("DeepSpeed did not load checkpoint client state")
    engine.train()

    dataset = Dataset.from_parquet(str(args.parquet.resolve()))
    if not 0 <= args.row_index < len(dataset):
        raise IndexError(f"row index {args.row_index} outside parquet with {len(dataset)} rows")
    batch = collate_fn([dataset[args.row_index]])
    batch = {name: value.to(engine.device) for name, value in batch.items()}
    context_tokens = int(batch["chunk_ids"].shape[1])
    context_nonpadding_tokens = int(batch["cross_attention_mask"].sum().item())
    query_tokens = int(batch["input_ids"].shape[1])

    engine.zero_grad()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = cuda_memory()
    started = time.monotonic()
    output = engine(**batch)
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"non-finite preflight loss: {loss}")
    after_forward = cuda_memory()
    engine.backward(loss)
    after_backward = cuda_memory()
    tracked_parameters = []
    for name, parameter in engine.module.named_parameters():
        gradient = parameter.grad
        if (
            parameter.requires_grad
            and parameter.numel() <= 4096
            and gradient is not None
            and bool(torch.isfinite(gradient).all().item())
            and bool(torch.any(gradient != 0).item())
        ):
            tracked_parameters.append((name, parameter, parameter.detach().clone()))
    if not tracked_parameters:
        raise RuntimeError("no small trainable parameter has a finite nonzero gradient")
    adam_steps_before = optimizer_state_steps(engine)
    engine_global_steps_before = int(engine.global_steps)
    # The optimizer state is already restored.  Calling the wrapped optimizer
    # directly exercises the AdamW update allocation without advancing or
    # saving the production checkpoint/scheduler/client cursor.
    engine.optimizer.step()
    adam_steps_after = optimizer_state_steps(engine)
    engine_global_steps_after = int(engine.global_steps)
    parameter_deltas = {
        name: float(
            (parameter.detach().float() - before.float()).abs().max().item()
        )
        for name, parameter, before in tracked_parameters
    }
    parameter_update_max_abs_delta = max(parameter_deltas.values())
    parameter_update_observed = any(delta > 0.0 for delta in parameter_deltas.values())
    optimizer_state_step_advanced = (
        len(adam_steps_before) == len(adam_steps_after)
        and all(after == before + 1 for before, after in zip(adam_steps_before, adam_steps_after))
    )
    after_optimizer = cuda_memory()
    elapsed = time.monotonic() - started

    rank_report: dict[str, object] = {
        "rank": rank,
        "device": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(),
        "loss": float(loss.detach()),
        "context_tokens": context_tokens,
        "context_nonpadding_tokens": context_nonpadding_tokens,
        "query_tokens": query_tokens,
        "engine_global_steps_before": engine_global_steps_before,
        "engine_global_steps_after": engine_global_steps_after,
        "adam_steps_before": adam_steps_before,
        "adam_steps_after": adam_steps_after,
        "optimizer_state_step_advanced": optimizer_state_step_advanced,
        "tracked_parameter_count": len(tracked_parameters),
        "parameter_update_max_abs_delta": parameter_update_max_abs_delta,
        "parameter_update_observed": parameter_update_observed,
        "elapsed_seconds": elapsed,
        "baseline": baseline,
        "after_forward": after_forward,
        "after_backward": after_backward,
        "after_optimizer": after_optimizer,
    }
    ranks = gather_rank_reports(rank_report)
    checks = capacity_checks(
        ranks,
        context_tokens=context_tokens,
        context_nonpadding_tokens=context_nonpadding_tokens,
        query_tokens=query_tokens,
    )
    if rank == 0:
        result = {
            "schema_version": 1,
            "scope": "real-checkpoint long-context forward/backward/AdamW capacity preflight",
            "invocation": invocation,
            "checkpoint": str(args.resume_root.resolve() / args.resume_tag),
            "optimizer_step": int(client_state["optimizer_step"]),
            "tensor_parallel_size": args.tensor_parallel_size,
            "parquet": str(args.parquet.resolve()),
            "parquet_bytes": args.parquet.stat().st_size,
            "parquet_sha256": sha256_file(args.parquet),
            "row_index": args.row_index,
            "context_tokens": context_tokens,
            "context_nonpadding_tokens": context_nonpadding_tokens,
            "query_tokens": query_tokens,
            "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
            "launcher_source_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "scripts/training/run_long_context_capacity_preflight.sh"
            ),
            "ranks": ranks,
            **checks,
        }
        write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if not checks["passed"]:
        raise RuntimeError("long-context capacity preflight failed its cross-rank checks")


if __name__ == "__main__":
    main()
