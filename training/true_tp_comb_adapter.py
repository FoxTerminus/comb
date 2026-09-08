#!/usr/bin/env python3
"""Experimental true tensor-parallel adapter for the official CombLlama.

This module is intentionally separate from the released model and the active
TP=1 reproduction path.  It fills the pieces that DeepSpeed 0.17.2 AutoTP
cannot infer from the custom Comb architecture:

* ``CombLlamaChunkModel.k_proj`` and ``v_proj`` live outside the decoder
  layers discovered by AutoTP, so they must be column-sharded explicitly.
* those projections produce the PIC K/V tensors, therefore the chunk model's
  local KV-head metadata must be updated with the same partition.
* cross-attention q/k RMSNorm weights are shared across heads.  Each TP rank
  sees only a subset of heads, so their partial gradients must be summed over
  the TP group before the optimizer step.

TP size one is a strict no-op.  The production trainer does not import this
file, which keeps the ongoing single-rank trajectory unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
import copy
import re

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.module_inject.layers import LinearLayer
from deepspeed.checkpoint.constants import (
    PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
    TP_REPLICATED_PARAMETER_PATTERNS,
    UNIVERSAL_CHECKPOINT_VERSION_KEY,
    UNIVERSAL_CHECKPOINT_VERSION_VALUE,
)
from deepspeed.utils import groups


def _sum_tp_gradient(group):
    def hook(gradient: torch.Tensor) -> torch.Tensor:
        summed = gradient.clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=group)
        return summed

    return hook


def _average_tp_gradient(group, tp_size: int):
    """Keep logically replicated gradients identical without changing scale."""

    def hook(gradient: torch.Tensor) -> torch.Tensor:
        averaged = gradient.clone()
        dist.all_reduce(averaged, op=dist.ReduceOp.SUM, group=group)
        averaged.div_(tp_size)
        return averaged

    return hook


def _shared_head_norm_parameters(model) -> Iterable[torch.nn.Parameter]:
    for layer in model.language_model.model.cross_layers:
        yield layer.cross_attn.q_norm.weight
        yield layer.cross_attn.k_norm.weight


_REPLICATED_PARAMETER = re.compile(
    r".*(?:input_layernorm|post_attention_layernorm|q_norm|k_norm|model\.norm)"
    r"\.weight$|.*cross_attn_(?:attn|mlp)_gate$"
)


def _other_trainable_replicated_parameters(
    model: torch.nn.Module,
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """Yield replicated parameters whose gradients are already full-sized.

    AutoTP's column-parallel backward returns a full input gradient, so layer
    norms and residual gates should receive the same logical gradient on every
    TP rank.  Large BF16 runs can nevertheless accumulate tiny rank-local
    rounding differences.  Averaging these gradients is scale preserving and
    makes the replicated-parameter invariant explicit.

    Cross-attention q/k norms are excluded because each rank sees different
    local heads; their partial gradients require the SUM hook above.
    """

    shared_head_norm_ids = {
        id(parameter) for parameter in _shared_head_norm_parameters(model)
    }
    for name, parameter in model.named_parameters():
        if (
            parameter.requires_grad
            and id(parameter) not in shared_head_norm_ids
            and _REPLICATED_PARAMETER.fullmatch(name)
        ):
            yield name, parameter


def _average_tensor_group_(
    tensors: list[torch.Tensor],
    group,
    tp_size: int,
    device: torch.device,
) -> None:
    """Average a small logical tensor group with one TP collective."""

    if not tensors:
        return
    flat = torch.cat(
        [tensor.detach().to(device=device, dtype=torch.float32).reshape(-1) for tensor in tensors]
    )
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    flat.div_(tp_size)
    offset = 0
    with torch.no_grad():
        for tensor in tensors:
            count = tensor.numel()
            restored = flat.narrow(0, offset, count).reshape(tensor.shape)
            tensor.copy_(restored.to(device=tensor.device, dtype=tensor.dtype))
            offset += count


def synchronize_true_tp_replicated_state(
    model: torch.nn.Module,
    tp_size: int,
) -> dict[str, int]:
    """Synchronize BF16 replicas and their ZeRO/Adam high-precision state.

    DeepSpeed AutoTP keeps layer norms and residual gates logically replicated,
    but ZeRO-2 CPU-offload can accumulate rank-local FP32/moment differences in
    a large BF16 model.  Gradient averaging alone cannot repair those hidden
    optimizer states.  Coalescing each state kind keeps the overhead to at most
    four small TP collectives per optimizer update.
    """

    if tp_size <= 1:
        return {}
    group = groups.get_tensor_model_parallel_group()
    device = next(model.parameters()).device
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _REPLICATED_PARAMETER.fullmatch(name)
    ]
    state_groups: dict[str, list[torch.Tensor]] = {
        "model_bf16": [parameter.data for parameter in parameters],
        "fp32_master": [],
        "exp_avg": [],
        "exp_avg_sq": [],
    }
    for parameter in parameters:
        mapping = getattr(parameter, "_hp_mapping", None)
        if mapping is None:
            continue
        state_groups["fp32_master"].append(mapping.get_hp_fragment())
        for state_name in ("exp_avg", "exp_avg_sq"):
            state = mapping.optim_fragment.get(state_name)
            if state is not None:
                state_groups[state_name].append(state)
    for tensors in state_groups.values():
        _average_tensor_group_(tensors, group, tp_size, device)
    return {name: len(tensors) for name, tensors in state_groups.items()}


def comb_universal_checkpoint_info() -> dict:
    """Describe Comb's TP layout to DeepSpeed's universal converter."""

    return {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        TP_REPLICATED_PARAMETER_PATTERNS: [
            r".*(?:input_layernorm|post_attention_layernorm|q_norm|k_norm|model.norm)\.weight$",
            r".*cross_attn_(?:attn|mlp)_gate$",
        ],
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [
            r".*(?:self_attn|cross_attn)\.o_proj\.weight$",
            r".*mlp\.down_proj\.weight$",
        ],
    }


def configure_deepspeed_for_true_tp(config: dict, tp_size: int) -> dict:
    """Return a config that preserves the logical batch across TP sizes.

    DeepSpeed validates and derives its micro batch using launcher world size,
    even though AutoTP ranks consume identical examples and the actual DP size
    is one.  Scaling the nominal global batch keeps the per-rank micro batch,
    gradient accumulation, and number of unique examples per update unchanged.
    """

    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    configured = copy.deepcopy(config)
    tensor_parallel = configured.setdefault("tensor_parallel", {})
    tensor_parallel["autotp_size"] = tp_size
    tensor_parallel.setdefault("tensor_parallel", {})["tp_size"] = tp_size
    if tp_size > 1:
        if "train_batch_size" not in configured:
            raise ValueError("true TP requires an explicit train_batch_size")
        configured["train_batch_size"] *= tp_size
    return configured


def apply_true_tp_comb_adapter(
    model: torch.nn.Module,
    tp_size: int,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Apply DeepSpeed AutoTP plus the missing Comb-specific partitions.

    The caller must initialize ``torch.distributed`` first.  For ``tp_size=1``
    this function returns the original model without replacing any module.
    """

    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if tp_size == 1:
        return model
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before true TP")
    if dist.get_world_size() % tp_size:
        raise ValueError(
            f"world size {dist.get_world_size()} is not divisible by TP size {tp_size}"
        )

    chunk_model = model.chunk_model
    global_kv_heads = int(chunk_model.num_key_value_heads)
    if global_kv_heads % tp_size:
        raise ValueError(
            f"chunk KV heads {global_kv_heads} are not divisible by TP size {tp_size}"
        )

    # AutoTP handles the standard LlamaDecoderLayer modules and the custom
    # cross-attention decoder layers.  It does not descend into the top-level
    # chunk K/V projection ModuleLists.
    deepspeed.tp_model_init(model, tp_size=tp_size, dtype=dtype)
    tp_group = groups.get_tensor_model_parallel_group()
    actual_tp_size = groups.get_tensor_model_parallel_world_size()
    if actual_tp_size != tp_size:
        raise RuntimeError(
            f"DeepSpeed created TP size {actual_tp_size}, expected {tp_size}"
        )

    for projection_name in ("k_proj", "v_proj"):
        projections = getattr(chunk_model, projection_name)
        for index, projection in enumerate(projections):
            if isinstance(projection, LinearLayer):
                raise RuntimeError(
                    f"chunk_model.{projection_name}.{index} was already TP-wrapped"
                )
            projections[index] = LinearLayer(
                projection,
                tp_group,
                name=f"chunk_model.{projection_name}.{index}",
            )

    chunk_model.num_key_value_heads = global_kv_heads // tp_size

    # q_norm/k_norm each have shape [head_dim], not [num_heads, head_dim], so
    # they remain replicated.  Their gradients are partial sums over local
    # heads and need a TP SUM (not an average) to match the unsharded model.
    shared_head_norms = list(_shared_head_norm_parameters(model))
    for parameter in shared_head_norms:
        if parameter.requires_grad:
            parameter.register_hook(_sum_tp_gradient(tp_group))

    averaged_replicas = list(_other_trainable_replicated_parameters(model))
    for _, parameter in averaged_replicas:
        parameter.register_hook(_average_tp_gradient(tp_group, tp_size))

    model._comb_true_tp = {
        "tp_size": tp_size,
        "global_chunk_kv_heads": global_kv_heads,
        "local_chunk_kv_heads": chunk_model.num_key_value_heads,
        "shared_head_norm_gradient_reduction": "sum",
        "other_replicated_gradient_reduction": "average",
        "averaged_replicated_parameters": [name for name, _ in averaged_replicas],
        "replicated_optimizer_state_sync": "coalesced_tp_average_after_update",
    }
    return model
