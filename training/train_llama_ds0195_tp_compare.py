#!/usr/bin/env python3
"""Controlled DeepSpeed 0.19.5 TP comparison for the official Comb recipe.

This entry point keeps the audited model, data order, loss, ZeRO-2, and
CPUAdam recipe unchanged.  It changes only tensor-parallel topology and uses
DeepSpeed's configuration-driven custom partitioning API.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from deepspeed.utils import groups
from safetensors.torch import save_file

from training import train_llama_repro as base


REQUIRED_DEEPSPEED = "0.19.5"
LOGICAL_BATCH_SIZE = 32
ACTIVE_ENGINE = None


def _cli_value(flag: str) -> str | None:
    for index, value in enumerate(sys.argv):
        if value == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def _sum_tp_gradient(group):
    def hook(gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    return hook


def _install_comb_shared_head_norm_hooks(model: torch.nn.Module, tp_size: int) -> int:
    """Reduce partial-head q/k norm gradients across the TP group."""
    if tp_size == 1:
        return 0
    group = groups.get_tensor_model_parallel_group()
    count = 0
    for layer in model.language_model.model.cross_layers:
        for parameter in (
            layer.cross_attn.q_norm.weight,
            layer.cross_attn.k_norm.weight,
        ):
            if parameter.requires_grad:
                parameter.register_hook(_sum_tp_gradient(group))
                count += 1
    return count


def _assert_partition(model: torch.nn.Module, tp_size: int) -> dict[str, int | bool]:
    parsed = bool(getattr(model, "ds_autotp_parsed", False))
    if tp_size > 1 and not parsed:
        raise RuntimeError("DeepSpeed did not apply AutoTP partitioning")

    global_heads = int(model.config.text_config.num_attention_heads)
    global_kv_heads = int(model.config.text_config.num_key_value_heads)
    expected_heads = global_heads // tp_size
    expected_kv_heads = global_kv_heads // tp_size
    cross = model.language_model.model.cross_layers[0].cross_attn
    chunk = model.chunk_model
    observed = {
        "autotp_parsed": parsed,
        "cross_local_heads": int(cross.num_heads),
        "cross_local_kv_heads": int(cross.num_key_value_heads),
        "chunk_local_kv_heads": int(chunk.num_key_value_heads),
        "expected_local_heads": expected_heads,
        "expected_local_kv_heads": expected_kv_heads,
    }
    if int(cross.num_heads) != expected_heads:
        raise RuntimeError(f"cross-attention head partition mismatch: {observed}")
    if int(cross.num_key_value_heads) != expected_kv_heads:
        raise RuntimeError(f"cross-attention KV-head partition mismatch: {observed}")
    if int(chunk.num_key_value_heads) != expected_kv_heads:
        raise RuntimeError(f"chunk KV-head partition mismatch: {observed}")
    return observed


def install_ds0195_tp_hooks() -> None:
    version = importlib.metadata.version("deepspeed")
    if version != REQUIRED_DEEPSPEED:
        raise RuntimeError(
            f"This comparison requires DeepSpeed {REQUIRED_DEEPSPEED}, found {version}"
        )

    original_initialize = base.deepspeed.initialize

    def initialize_for_comparison(*args, **kwargs):
        global ACTIVE_ENGINE
        if args:
            raise TypeError("comparison wrapper requires keyword DeepSpeed initialization")
        model = kwargs["model"]
        original = kwargs["config"]
        tp_size = int(original["tensor_parallel"]["autotp_size"])
        if dist.get_world_size() != tp_size:
            raise ValueError("comparison arms require pure TP: world_size must equal tp_size")

        configured = copy.deepcopy(original)
        configured["train_batch_size"] = LOGICAL_BATCH_SIZE * tp_size
        configured["tensor_parallel"]["autotp_size"] = tp_size
        configured["tensor_parallel"]["tp"] = {"tp_size": tp_size}
        configured["tensor_parallel"].pop("tensor_parallel", None)

        output_value = _cli_value("--output-dir")
        if output_value is None:
            raise ValueError("--output-dir is required")
        output_dir = Path(output_value).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        configured.setdefault("tensorboard", {})["output_path"] = str(output_dir)

        kwargs["config"] = configured
        result = original_initialize(**kwargs)
        engine = result[0]
        ACTIVE_ENGINE = engine
        if int(engine.dp_world_size) != 1:
            raise RuntimeError(f"expected DP size 1, got {engine.dp_world_size}")
        if int(engine.train_micro_batch_size_per_gpu()) != 8:
            raise RuntimeError(
                "TP changed the logical micro batch: "
                f"observed {engine.train_micro_batch_size_per_gpu()}, expected 8"
            )

        partition = _assert_partition(engine.module, tp_size)
        hook_count = _install_comb_shared_head_norm_hooks(engine.module, tp_size)
        if dist.get_rank() == 0:
            manifest = {
                "created_unix": time.time(),
                "deepspeed": version,
                "tp_size": tp_size,
                "dp_size": int(engine.dp_world_size),
                "logical_train_batch_size": LOGICAL_BATCH_SIZE,
                "deepspeed_nominal_train_batch_size": configured["train_batch_size"],
                "micro_batch_size": int(engine.train_micro_batch_size_per_gpu()),
                "gradient_accumulation_steps": int(engine.gradient_accumulation_steps()),
                "zero_stage": int(engine.zero_optimization_stage()),
                "optimizer_class": type(engine.optimizer).__name__,
                "shared_head_norm_hooks": hook_count,
                "partition": partition,
                "config": configured,
            }
            (output_dir / "ds0195_tp_runtime_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(json.dumps(manifest), flush=True)
        return result

    base.deepspeed.initialize = initialize_for_comparison


def export_trainable_state(output_dir: Path) -> None:
    """Save only trainable BF16 parameters, excluding large optimizer state."""
    if ACTIVE_ENGINE is None:
        raise RuntimeError("DeepSpeed engine was not initialized")
    engine = ACTIVE_ENGINE
    trainable_names = {
        name for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    # ZeRO stages 0-2 replicate parameters across data-parallel ranks.  With
    # TP=1 there is therefore nothing for DeepSpeed to consolidate, and
    # _consolidated_16bit_state_dict() intentionally raises.  TP>1 still has
    # model-parallel parameter shards and must use DeepSpeed's consolidation
    # path (as must ZeRO-3, which partitions parameters across DP ranks).
    tp_size = int(_cli_value("--tensor-parallel-size") or "1")
    zero_stage = int(engine.zero_optimization_stage())
    if tp_size == 1 and zero_stage < 3:
        state = engine.module.state_dict()
    else:
        state = engine._consolidated_16bit_state_dict()
    if dist.get_rank() == 0:
        missing = sorted(trainable_names.difference(state))
        if missing:
            raise RuntimeError(
                f"consolidated state is missing {len(missing)} trainable parameters: "
                f"{missing[:8]}"
            )
        trainable = {
            name: state[name].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            for name in sorted(trainable_names)
        }
        path = output_dir / "trainable_final.safetensors"
        save_file(trainable, path)
        manifest = {
            "format": "consolidated_trainable_bf16",
            "parameter_tensors": len(trainable),
            "parameter_elements": sum(tensor.numel() for tensor in trainable.values()),
            "bytes": path.stat().st_size,
            "path": path.name,
        }
        (output_dir / "trainable_final_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps(manifest), flush=True)
    del state
    dist.barrier()


def main() -> None:
    install_ds0195_tp_hooks()
    original_destroy = base.torch.distributed.destroy_process_group
    base.torch.distributed.destroy_process_group = lambda: None
    try:
        base.main()
        if os.environ.get("COMB_TP_COMPARE_EXPORT_TRAINABLE") == "1":
            output_value = _cli_value("--output-dir")
            if output_value is None:
                raise ValueError("--output-dir is required")
            export_trainable_state(Path(output_value).resolve())
    finally:
        base.torch.distributed.destroy_process_group = original_destroy
        if dist.is_initialized():
            original_destroy()


if __name__ == "__main__":
    main()
