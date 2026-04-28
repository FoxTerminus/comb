"""Megatron-style tensor parallel adaptation for SambaY-Llama.

This mirrors the local YOCO TP adapter while accounting for SambaY's extra
Mamba/GMU memory branch. The important difference is the Mamba ``in_proj``:
it emits ``[gate, value]`` with both halves sized ``d_mem``. A naive output
shard would give rank 0 mostly gate channels and rank 1 mostly value channels,
so this file uses a gated-pair column split that preserves both halves locally.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from models.CombLlama_megatron import ColumnParallelLinear, RowParallelLinear, copy_to_tensor_parallel_region


class GatedPairColumnParallelLinear(nn.Module):
    """Column-parallel Linear for concatenated ``[gate, value]`` projections."""

    def __init__(self, in_features: int, half_out_features: int, tp_group: dist.ProcessGroup, bias: bool = False):
        super().__init__()
        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        if half_out_features % self.tp_size != 0:
            raise ValueError(
                f"half_out_features={half_out_features} must be divisible by tp_size={self.tp_size}"
            )
        self.in_features = in_features
        self.half_out_features = half_out_features
        self.half_out_features_per_rank = half_out_features // self.tp_size
        self.out_features = 2 * half_out_features
        self.weight = nn.Parameter(torch.empty(2 * self.half_out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(2 * self.half_out_features_per_rank))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            x = copy_to_tensor_parallel_region(x, self.tp_group)
        return nn.functional.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, tp_group: dist.ProcessGroup) -> "GatedPairColumnParallelLinear":
        if linear.out_features % 2 != 0:
            raise ValueError(f"Expected an even gated-pair out_features, got {linear.out_features}")
        half = linear.out_features // 2
        tp_rank = dist.get_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)
        chunk = half // tp_size
        layer = cls(linear.in_features, half, tp_group, bias=linear.bias is not None)
        with torch.no_grad():
            gate = linear.weight.data[tp_rank * chunk : (tp_rank + 1) * chunk]
            value = linear.weight.data[half + tp_rank * chunk : half + (tp_rank + 1) * chunk]
            layer.weight.copy_(torch.cat([gate, value], dim=0))
            if linear.bias is not None:
                gate_bias = linear.bias.data[tp_rank * chunk : (tp_rank + 1) * chunk]
                value_bias = linear.bias.data[half + tp_rank * chunk : half + (tp_rank + 1) * chunk]
                layer.bias.copy_(torch.cat([gate_bias, value_bias], dim=0))
        return layer


def _replace_attention_linear(attn_module, tp_group, column_names, row_names):
    for name in column_names:
        old = getattr(attn_module, name)
        new = ColumnParallelLinear.from_linear(old, tp_group).to(device=old.weight.device, dtype=old.weight.dtype)
        setattr(attn_module, name, new)
    for name in row_names:
        old = getattr(attn_module, name)
        new = RowParallelLinear.from_linear(old, tp_group).to(device=old.weight.device, dtype=old.weight.dtype)
        setattr(attn_module, name, new)


def _replace_mlp_linear(mlp_module, tp_group):
    gate_old = mlp_module.gate_proj
    up_old = mlp_module.up_proj
    down_old = mlp_module.down_proj
    mlp_module.gate_proj = ColumnParallelLinear.from_linear(gate_old, tp_group).to(
        device=gate_old.weight.device,
        dtype=gate_old.weight.dtype,
    )
    mlp_module.up_proj = ColumnParallelLinear.from_linear(up_old, tp_group).to(
        device=up_old.weight.device,
        dtype=up_old.weight.dtype,
    )
    mlp_module.down_proj = RowParallelLinear.from_linear(down_old, tp_group).to(
        device=down_old.weight.device,
        dtype=down_old.weight.dtype,
    )


def _patch_attention_heads(attn_module, tp_size):
    attn_module._tp_size = tp_size
    if attn_module.num_heads % tp_size != 0:
        raise ValueError(f"num_heads={attn_module.num_heads} is not divisible by tp_size={tp_size}")
    if attn_module.num_key_value_heads % tp_size != 0:
        raise ValueError(
            f"num_key_value_heads={attn_module.num_key_value_heads} is not divisible by tp_size={tp_size}"
        )
    attn_module.num_heads //= tp_size
    attn_module.num_key_value_heads //= tp_size
    attn_module.kv_repeats = attn_module.num_heads // attn_module.num_key_value_heads


def _replace_mamba_linear(mamba_module, tp_group):
    in_old = mamba_module.in_proj
    decay_old = mamba_module.decay_proj
    out_old = mamba_module.out_proj
    mamba_module.in_proj = GatedPairColumnParallelLinear.from_linear(in_old, tp_group).to(
        device=in_old.weight.device,
        dtype=in_old.weight.dtype,
    )
    mamba_module.decay_proj = ColumnParallelLinear.from_linear(decay_old, tp_group).to(
        device=decay_old.weight.device,
        dtype=decay_old.weight.dtype,
    )
    mamba_module.out_proj = RowParallelLinear.from_linear(out_old, tp_group).to(
        device=out_old.weight.device,
        dtype=out_old.weight.dtype,
    )
    mamba_module.d_mem //= dist.get_world_size(tp_group)
    # Official Mamba internals are not sharded by this adapter. Use the local
    # recurrence path under TP so the GMU memory shard and gradients are valid.
    mamba_module.official_mamba = None
    mamba_module.uses_official_mamba = False


def _replace_gmu_linear(gmu_module, tp_group):
    in_old = gmu_module.in_proj
    out_old = gmu_module.out_proj
    gmu_module.in_proj = ColumnParallelLinear.from_linear(in_old, tp_group).to(
        device=in_old.weight.device,
        dtype=in_old.weight.dtype,
    )
    gmu_module.out_proj = RowParallelLinear.from_linear(out_old, tp_group).to(
        device=out_old.weight.device,
        dtype=out_old.weight.dtype,
    )


def _apply_to_sambay_layer(layer, tp_group, tp_size):
    if layer.use_mamba:
        _replace_mamba_linear(layer.token_mixer, tp_group)
    else:
        _replace_attention_linear(
            layer.token_mixer,
            tp_group,
            column_names=["q_proj", "k_proj", "v_proj"],
            row_names=["o_proj"],
        )
        _patch_attention_heads(layer.token_mixer, tp_size)
    _replace_mlp_linear(layer.mlp, tp_group)


def apply_tensor_parallelism(model, tp_group: dist.ProcessGroup):
    """Apply tensor parallelism to ``SambaYForCausalLM`` after loading weights."""

    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return

    text_model = model.model

    for layer in text_model.self_decoder.layers:
        _apply_to_sambay_layer(layer, tp_group, tp_size)

    _apply_to_sambay_layer(text_model.self_decoder.gmu_save_layer, tp_group, tp_size)

    boundary_attn = text_model.self_decoder.boundary_layer.self_attn
    _replace_attention_linear(
        boundary_attn,
        tp_group,
        column_names=["q_proj", "k_proj", "v_proj"],
        row_names=["o_proj"],
    )
    _patch_attention_heads(boundary_attn, tp_size)
    _replace_mlp_linear(text_model.self_decoder.boundary_layer.mlp, tp_group)

    for layer in text_model.cross_decoder.layers:
        if layer.use_gmu:
            _replace_gmu_linear(layer.token_mixer, tp_group)
        else:
            _replace_attention_linear(
                layer.token_mixer,
                tp_group,
                column_names=["q_proj"],
                row_names=["o_proj"],
            )
            _patch_attention_heads(layer.token_mixer, tp_size)
        _replace_mlp_linear(layer.mlp, tp_group)

    lm_head = model.lm_head
    model.lm_head = ColumnParallelLinear.from_linear(lm_head, tp_group, gather_output=True).to(
        device=lm_head.weight.device,
        dtype=lm_head.weight.dtype,
    )
