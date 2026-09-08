#!/usr/bin/env python3
"""End-to-end native checkpoint -> Universal -> true-TP acceptance test.

The source mode creates a tiny checkpoint after deterministic training
and records the full trainable state.  The resume mode loads the converted
universal checkpoint into a true-TP model and verifies both the restored state
and the following optimizer update against the unsharded reference trajectory.
The reshard mode additionally validates native TP=2 -> Universal -> TP=4.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.checkpoint.constants import (
    UNIVERSAL_CHECKPOINT_INFO,
    UNIVERSAL_CHECKPOINT_VERSION_KEY,
)

from training.true_tp_comb_adapter import (
    apply_true_tp_comb_adapter,
    comb_universal_checkpoint_info,
    configure_deepspeed_for_true_tp,
)
from training.validate_true_tp_comb_tiny import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("source", "resume", "native-resume", "reshard-resume"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-steps", type=int, default=3)
    parser.add_argument("--model-kv-heads", type=int, default=2)
    parser.add_argument(
        "--report",
        type=Path,
        help="On rank zero, atomically write the structured acceptance result.",
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(0, 2),
        default=2,
        help="Use stage 0 to mirror the production TP checkpoint fallback.",
    )
    return parser.parse_args()


def emit_report(args: argparse.Namespace, rank: int, payload) -> None:
    if rank != 0:
        return
    print(json.dumps(payload, indent=2), flush=True)
    if args.report is None:
        return
    report = {
        "mode": args.mode,
        "zero_stage": args.zero_stage,
        "model_kv_heads": args.model_kv_heads,
        "results": payload,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.report)


def ds_config(
    tp_size: int, load_universal: bool = False, zero_stage: int = 2
) -> dict:
    config = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-3, "weight_decay": 0.01},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 1e-3,
                "warmup_num_steps": 0,
                "total_num_steps": 100,
            },
        },
        "zero_optimization": {"stage": zero_stage},
    }
    if zero_stage == 2:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    return configure_deepspeed_for_true_tp(config, tp_size)


def deterministic_batch(device: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(3, 120, (2, 8), generator=generator).to(device)
    chunk_ids = torch.randint(3, 120, (2, 8), generator=generator).to(device)
    return {
        "input_ids": input_ids,
        "chunk_ids": chunk_ids,
        "attention_mask": torch.ones_like(input_ids),
        "cross_attention_mask": torch.ones_like(chunk_ids),
        "use_cache": False,
    }


def train_step(engine, batch: dict[str, torch.Tensor]) -> float:
    output = engine(**batch)
    loss = output.logits.float().square().mean()
    engine.backward(loss)
    engine.step()
    return float(loss.detach())


def bf16_loss_tolerance(reference: float) -> float:
    """Allow reduction-order rounding, while rejecting material divergence."""

    return max(2e-5, abs(reference) * 1e-3)


def bf16_reshard_loss_tolerance(reference: float) -> float:
    """Allow two TP reduction-order changes around a universal conversion."""

    return max(4e-5, abs(reference) * 2e-3)


def trainable_state(module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def expected_local(full: torch.Tensor, local: torch.Tensor, rank: int, world: int):
    if full.shape == local.shape:
        return full
    shard_dims = [
        dim
        for dim in range(full.ndim)
        if full.shape[dim] == local.shape[dim] * world
        and all(
            full.shape[other] == local.shape[other]
            for other in range(full.ndim)
            if other != dim
        )
    ]
    if len(shard_dims) != 1:
        raise AssertionError((full.shape, local.shape, shard_dims))
    return torch.chunk(full, world, dim=shard_dims[0])[rank]


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(42)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(num_key_value_heads=args.model_kv_heads)
    apply_true_tp_comb_adapter(model, tp_size=world, dtype=torch.bfloat16)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config(
            world,
            load_universal=args.mode in {"resume", "reshard-resume"},
            zero_stage=args.zero_stage,
        ),
    )
    engine.train()
    batch = deterministic_batch(local_rank)

    if args.mode == "source":
        if world != 1:
            raise ValueError("source mode requires world size one")
        losses = [train_step(engine, batch) for _ in range(args.source_steps)]
        engine.save_checkpoint(
            str(args.work_dir / "source"),
            tag="step_source",
            client_state={
                "source_steps": args.source_steps,
                UNIVERSAL_CHECKPOINT_INFO: comb_universal_checkpoint_info(),
            },
        )
        torch.save(
            {
                "at_checkpoint": trainable_state(engine.module),
                "global_steps": engine.global_steps,
                "lr_scheduler": engine.lr_scheduler.state_dict(),
            },
            args.work_dir / "reference_checkpoint.pt",
        )
        next_loss = train_step(engine, batch)
        torch.save(
            {
                "after_next_step": trainable_state(engine.module),
                "next_loss": next_loss,
                "global_steps": engine.global_steps,
                "lr_scheduler": engine.lr_scheduler.state_dict(),
            },
            args.work_dir / "reference_next_step.pt",
        )
        following_loss = train_step(engine, batch)
        torch.save(
            {
                "after_following_step": trainable_state(engine.module),
                "following_loss": following_loss,
                "global_steps": engine.global_steps,
                "lr_scheduler": engine.lr_scheduler.state_dict(),
            },
            args.work_dir / "reference_following_step.pt",
        )
        emit_report(
            args,
            rank,
            {
                "source_losses": losses,
                "next_loss": next_loss,
                "following_loss": following_loss,
            },
        )
    elif args.mode == "resume":
        if world < 2:
            raise ValueError("resume mode requires true TP world size greater than one")
        load_path, client_state = engine.load_checkpoint(str(args.work_dir / "universal"))
        if load_path is None:
            raise RuntimeError("universal checkpoint load failed")
        checkpoint_reference = torch.load(
            args.work_dir / "reference_checkpoint.pt", map_location="cpu", weights_only=False
        )
        before_deltas = {}
        for name, parameter in engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            expected = expected_local(
                checkpoint_reference["at_checkpoint"][name], parameter, rank, world
            )
            before_deltas[name] = float(
                (expected - parameter.detach().float().cpu()).abs().max()
            )

        next_loss = train_step(engine, batch)
        next_reference = torch.load(
            args.work_dir / "reference_next_step.pt", map_location="cpu", weights_only=False
        )
        after_deltas = {}
        for name, parameter in engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            expected = expected_local(
                next_reference["after_next_step"][name], parameter, rank, world
            )
            after_deltas[name] = float(
                (expected - parameter.detach().float().cpu()).abs().max()
            )
        report = {
            "rank": rank,
            "world": world,
            "client_source_steps": client_state["source_steps"],
            "universal_checkpoint_version": client_state[
                UNIVERSAL_CHECKPOINT_INFO
            ][UNIVERSAL_CHECKPOINT_VERSION_KEY],
            "loaded_global_steps": checkpoint_reference["global_steps"],
            "engine_global_steps_after_next": engine.global_steps,
            "reference_global_steps_after_next": next_reference["global_steps"],
            "max_parameter_delta_before_next": max(before_deltas.values()),
            "next_loss": next_loss,
            "reference_next_loss": next_reference["next_loss"],
            "next_loss_abs_delta": abs(next_loss - next_reference["next_loss"]),
            "next_loss_tolerance": bf16_loss_tolerance(next_reference["next_loss"]),
            "max_parameter_delta_after_next": max(after_deltas.values()),
            "lr_scheduler_matches": (
                engine.lr_scheduler.state_dict() == next_reference["lr_scheduler"]
            ),
        }
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, report)
        assert report["max_parameter_delta_before_next"] == 0.0
        assert engine.global_steps == next_reference["global_steps"]
        assert report["lr_scheduler_matches"]
        assert abs(next_loss - next_reference["next_loss"]) < bf16_loss_tolerance(
            next_reference["next_loss"]
        )
        assert report["max_parameter_delta_after_next"] < 2e-3

        # Preserve the actual rank-local TP state.  It may legitimately differ
        # slightly from the unsharded BF16 reference after an optimizer step;
        # native restart fidelity must be measured against this state instead.
        torch.save(
            trainable_state(engine.module),
            args.work_dir / f"native_tp_rank_{rank}_before_save.pt",
        )
        dist.barrier()

        engine.save_checkpoint(
            str(args.work_dir / "native_tp"),
            tag="step_native",
            client_state={"source_steps": args.source_steps},
        )
        dist.barrier()
        emit_report(args, rank, gathered)
    elif args.mode == "native-resume":
        if world < 2:
            raise ValueError("native-resume mode requires true TP world size greater than one")
        load_path, client_state = engine.load_checkpoint(
            str(args.work_dir / "native_tp"), tag="step_native"
        )
        if load_path is None:
            raise RuntimeError("native TP checkpoint load failed")
        before_reference = torch.load(
            args.work_dir / f"native_tp_rank_{rank}_before_save.pt",
            map_location="cpu",
            weights_only=False,
        )
        before_deltas = {}
        for name, parameter in engine.module.named_parameters():
            if parameter.requires_grad:
                expected = before_reference[name]
                before_deltas[name] = float(
                    (expected - parameter.detach().float().cpu()).abs().max()
                )
        following_loss = train_step(engine, batch)
        following_reference = torch.load(
            args.work_dir / "reference_following_step.pt",
            map_location="cpu",
            weights_only=False,
        )
        after_deltas = {}
        for name, parameter in engine.module.named_parameters():
            if parameter.requires_grad:
                expected = expected_local(
                    following_reference["after_following_step"][name],
                    parameter,
                    rank,
                    world,
                )
                after_deltas[name] = float(
                    (expected - parameter.detach().float().cpu()).abs().max()
                )
        report = {
            "rank": rank,
            "world": world,
            "client_source_steps": client_state["source_steps"],
            "max_parameter_delta_after_native_load": max(before_deltas.values()),
            "following_loss": following_loss,
            "reference_following_loss": following_reference["following_loss"],
            "following_loss_abs_delta": abs(
                following_loss - following_reference["following_loss"]
            ),
            "following_loss_tolerance": bf16_loss_tolerance(
                following_reference["following_loss"]
            ),
            "max_parameter_delta_after_following_step": max(after_deltas.values()),
            "engine_global_steps": engine.global_steps,
            "reference_global_steps": following_reference["global_steps"],
            "lr_scheduler_matches": (
                engine.lr_scheduler.state_dict()
                == following_reference["lr_scheduler"]
            ),
        }
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, report)
        assert report["max_parameter_delta_after_native_load"] == 0.0
        assert abs(
            following_loss - following_reference["following_loss"]
        ) < bf16_loss_tolerance(following_reference["following_loss"])
        assert report["max_parameter_delta_after_following_step"] < 2e-3
        assert engine.global_steps == following_reference["global_steps"]
        assert report["lr_scheduler_matches"]
        dist.barrier()
        emit_report(args, rank, gathered)

    else:
        if world < 2:
            raise ValueError("reshard-resume mode requires true TP world size greater than one")
        load_path, client_state = engine.load_checkpoint(
            str(args.work_dir / "universal_from_native")
        )
        if load_path is None:
            raise RuntimeError("converted native TP universal checkpoint load failed")

        checkpoint_reference = torch.load(
            args.work_dir / "reference_next_step.pt",
            map_location="cpu",
            weights_only=False,
        )
        native_reference = None
        if world == 2:
            native_reference = torch.load(
                args.work_dir / f"native_tp_rank_{rank}_before_save.pt",
                map_location="cpu",
                weights_only=False,
            )
        before_deltas = {}
        native_before_deltas = {}
        for name, parameter in engine.module.named_parameters():
            if parameter.requires_grad:
                expected = expected_local(
                    checkpoint_reference["after_next_step"][name],
                    parameter,
                    rank,
                    world,
                )
                before_deltas[name] = float(
                    (expected - parameter.detach().float().cpu()).abs().max()
                )
                if native_reference is not None:
                    native_before_deltas[name] = float(
                        (
                            native_reference[name]
                            - parameter.detach().float().cpu()
                        )
                        .abs()
                        .max()
                    )

        following_loss = train_step(engine, batch)
        following_reference = torch.load(
            args.work_dir / "reference_following_step.pt",
            map_location="cpu",
            weights_only=False,
        )
        after_deltas = {}
        for name, parameter in engine.module.named_parameters():
            if parameter.requires_grad:
                expected = expected_local(
                    following_reference["after_following_step"][name],
                    parameter,
                    rank,
                    world,
                )
                after_deltas[name] = float(
                    (expected - parameter.detach().float().cpu()).abs().max()
                )
        report = {
            "rank": rank,
            "world": world,
            "client_source_steps": client_state["source_steps"],
            "max_parameter_delta_after_reshard_load": max(before_deltas.values()),
            "max_parameter_delta_from_native_tp_source": (
                max(native_before_deltas.values()) if native_before_deltas else None
            ),
            "following_loss": following_loss,
            "reference_following_loss": following_reference["following_loss"],
            "following_loss_abs_delta": abs(
                following_loss - following_reference["following_loss"]
            ),
            "following_loss_tolerance": bf16_reshard_loss_tolerance(
                following_reference["following_loss"]
            ),
            "max_parameter_delta_after_following_step": max(after_deltas.values()),
            "engine_global_steps": engine.global_steps,
            "reference_global_steps": following_reference["global_steps"],
            "lr_scheduler_matches": (
                engine.lr_scheduler.state_dict()
                == following_reference["lr_scheduler"]
            ),
        }
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, report)
        assert report["max_parameter_delta_after_reshard_load"] < 2e-3
        if native_reference is not None:
            assert report["max_parameter_delta_from_native_tp_source"] == 0.0
        assert abs(
            following_loss - following_reference["following_loss"]
        ) < bf16_reshard_loss_tolerance(following_reference["following_loss"])
        assert report["max_parameter_delta_after_following_step"] < 2e-3
        assert engine.global_steps == following_reference["global_steps"]
        assert report["lr_scheduler_matches"]
        dist.barrier()
        emit_report(args, rank, gathered)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
