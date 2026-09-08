#!/usr/bin/env python3
"""Train the frozen-32-layer PIC model on the official Comb data format.

The first controlled run intentionally uses only official SQuAD.  Each sample's
complete context remains one PIC chunk; multi-document splitting is deferred to
a later experiment.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from deepspeed.utils import groups

from comb.integration.hf.CombLlamaFrozen32 import (
    CombLlamaFrozen32Config,
    CombLlamaFrozen32ForConditionalGeneration,
)
from training import train_llama_repro as base


REQUIRED_DEEPSPEED = "0.19.5"
LOGICAL_BATCH_SIZE = 32
INITIAL_DATASETS = [
    name.strip()
    for name in os.environ.get("FROZEN32_DATASETS", "SQuAD").split(",")
    if name.strip()
]
UNKNOWN_DATASETS = sorted(set(INITIAL_DATASETS).difference(base.DATASET_DICT))
if not INITIAL_DATASETS or UNKNOWN_DATASETS:
    raise ValueError(f"invalid FROZEN32_DATASETS: {UNKNOWN_DATASETS or INITIAL_DATASETS}")


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


def _install_partial_head_norm_hooks(model: torch.nn.Module, tp_size: int) -> int:
    if tp_size == 1:
        return 0
    group = groups.get_tensor_model_parallel_group()
    count = 0
    for layer in model.language_model.model.cross_layers:
        for parameter in (layer.cross_attn.q_norm.weight, layer.cross_attn.k_norm.weight):
            parameter.register_hook(_sum_tp_gradient(group))
            count += 1
    return count


def _assert_frozen32_partition(model: torch.nn.Module, tp_size: int) -> dict:
    parsed = bool(getattr(model, "ds_autotp_parsed", False))
    if tp_size > 1 and not parsed:
        raise RuntimeError("DeepSpeed did not apply AutoTP partitioning")
    global_heads = int(model.config.text_config.num_attention_heads)
    global_kv_heads = int(model.config.text_config.num_key_value_heads)
    cross = model.language_model.model.cross_layers[0].cross_attn
    expected_heads = global_heads // tp_size
    expected_kv_heads = global_kv_heads // tp_size
    observed = {
        "autotp_parsed": parsed,
        "cross_local_heads": int(cross.num_heads),
        "cross_local_kv_heads": int(cross.num_key_value_heads),
        "expected_local_heads": expected_heads,
        "expected_local_kv_heads": expected_kv_heads,
        "q_proj_shape": list(cross.q_proj.weight.shape),
        "k_proj_shape": list(cross.k_proj.weight.shape),
        "o_proj_shape": list(cross.o_proj.weight.shape),
    }
    if int(cross.num_heads) != expected_heads:
        raise RuntimeError(f"cross Q-head partition mismatch: {observed}")
    if int(cross.num_key_value_heads) != expected_kv_heads:
        raise RuntimeError(f"cross KV-head partition mismatch: {observed}")

    illegal_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("language_model.model.cross_layers.")
    ]
    if illegal_trainable:
        raise RuntimeError(f"frozen parameter boundary violated: {illegal_trainable[:8]}")
    return observed


def _build_model(*, from_scratch: bool, config: CombLlamaFrozen32Config):
    if not from_scratch:
        raise ValueError("the frozen32 training entry requires pretrained initialization")
    return CombLlamaFrozen32ForConditionalGeneration.from_llama_pretrained(
        base.MODEL_NAME,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def install_training_hooks() -> None:
    version = importlib.metadata.version("deepspeed")
    if version != REQUIRED_DEEPSPEED:
        raise RuntimeError(f"requires DeepSpeed {REQUIRED_DEEPSPEED}, found {version}")

    base.CombLlamaConfig = CombLlamaFrozen32Config
    base.CombLlamaForConditionalGeneration = _build_model
    base.TRAIN_DATASETS = list(INITIAL_DATASETS)
    base.ADDITIONAL_SOURCE_SNAPSHOT_FILES = {
        "training/train_llama_frozen32.py": Path(__file__).resolve(),
        "data/NaturalInstructions.py": (
            Path(__file__).resolve().parents[1] / "data/NaturalInstructions.py"
        ),
        "data/curate_ni.py": (
            Path(__file__).resolve().parents[1] / "data/curate_ni.py"
        ),
        "comb/integration/hf/CombLlamaFrozen32.py": (
            Path(__file__).resolve().parents[1]
            / "comb/integration/hf/CombLlamaFrozen32.py"
        ),
    }
    original_initialize = base.deepspeed.initialize

    def initialize_frozen32(*args, **kwargs):
        if args:
            raise TypeError("frozen32 wrapper requires keyword DeepSpeed initialization")
        original_config = kwargs["config"]
        tp_size = int(original_config["tensor_parallel"]["autotp_size"])
        if dist.get_world_size() != tp_size:
            raise ValueError("the first frozen32 run requires pure TP (DP size one)")

        configured = copy.deepcopy(original_config)
        # DeepSpeed validates this field against launcher world size before it
        # constructs the TP mesh.  Logical batch remains 8 * 4 = 32 at DP=1.
        configured["train_batch_size"] = LOGICAL_BATCH_SIZE * tp_size
        configured["tensor_parallel"]["tp"] = {"tp_size": tp_size}
        configured["tensor_parallel"].pop("tensor_parallel", None)
        kwargs["config"] = configured
        result = original_initialize(**kwargs)
        engine = result[0]
        if int(engine.dp_world_size) != 1:
            raise RuntimeError(f"expected DP size 1, got {engine.dp_world_size}")
        if int(engine.zero_optimization_stage()) != 0:
            raise RuntimeError("the first frozen32 run requires ZeRO Stage 0")
        partition = _assert_frozen32_partition(engine.module, tp_size)
        hook_count = _install_partial_head_norm_hooks(engine.module, tp_size)

        # Frozen encoder/decoder weights are reproducibly reloaded from the
        # Llama backbone on resume.  Keeping them in every TP checkpoint wastes
        # tens of GB and would violate the repository's disk-space budget.
        original_save_checkpoint = engine.save_checkpoint
        original_load_checkpoint = engine.load_checkpoint

        def save_trainable_checkpoint(*save_args, **save_kwargs):
            save_kwargs.setdefault("exclude_frozen_parameters", True)
            return original_save_checkpoint(*save_args, **save_kwargs)

        def load_trainable_checkpoint(*load_args, **load_kwargs):
            load_kwargs.setdefault("load_module_strict", False)
            return original_load_checkpoint(*load_args, **load_kwargs)

        engine.save_checkpoint = save_trainable_checkpoint
        engine.load_checkpoint = load_trainable_checkpoint

        if dist.get_rank() == 0:
            output_dir = Path(_cli_value("--output-dir")).resolve()
            runtime = {
                "architecture": "frozen 32-layer encoder + frozen 32-layer decoder",
                "cross_attention_layers": engine.module.config.cross_attention_layers,
                "trainable_scope": "eight cross-attention + context-only MLP branches",
                "datasets": INITIAL_DATASETS,
                "deepspeed": version,
                "tp_size": tp_size,
                "dp_size": int(engine.dp_world_size),
                "zero_stage": int(engine.zero_optimization_stage()),
                "logical_batch_size": LOGICAL_BATCH_SIZE,
                "deepspeed_nominal_batch_size": configured["train_batch_size"],
                "optimizer": type(engine.optimizer).__name__,
                "checkpoint_excludes_frozen_parameters": True,
                "partial_head_norm_hooks": hook_count,
                "partition": partition,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "frozen32_runtime_manifest.json").write_text(
                json.dumps(runtime, indent=2) + "\n"
            )
            print(json.dumps(runtime), flush=True)
        return result

    base.deepspeed.initialize = initialize_frozen32


def main() -> None:
    install_training_hooks()
    base.main()
    local_rank = int(base.os.environ.get("LOCAL_RANK", "0"))
    print(
        json.dumps(
            {
                "local_rank": local_rank,
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / (1024**3), 3
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
