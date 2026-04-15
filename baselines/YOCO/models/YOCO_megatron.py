"""Megatron-style tensor parallel adaptation for the YOCO-Llama baseline."""

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from models.CombLlama_megatron import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from baselines.YOCO.models.YOCO import _apply_rotary_to_single


def _replace_attention_linear(attn_module, tp_group, column_names, row_names):
    for name in column_names:
        old = getattr(attn_module, name)
        setattr(attn_module, name, ColumnParallelLinear.from_linear(old, tp_group))
    for name in row_names:
        old = getattr(attn_module, name)
        setattr(attn_module, name, RowParallelLinear.from_linear(old, tp_group))


def _replace_mlp_linear(mlp_module, tp_group):
    mlp_module.gate_proj = ColumnParallelLinear.from_linear(mlp_module.gate_proj, tp_group)
    mlp_module.up_proj = ColumnParallelLinear.from_linear(mlp_module.up_proj, tp_group)
    mlp_module.down_proj = RowParallelLinear.from_linear(mlp_module.down_proj, tp_group)


def _patch_num_heads(module, tp_size, head_attrs):
    module._tp_size = tp_size
    for attr in head_attrs:
        if hasattr(module, attr):
            setattr(module, attr, getattr(module, attr) // tp_size)


def apply_tensor_parallelism(model, tp_group: dist.ProcessGroup):
    """Apply tensor parallelism to a YOCOForCausalLM model."""
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return

    text_model = model.model

    # 1. Self-decoder SWA layers
    for layer in text_model.self_decoder.layers:
        attn = layer.self_attn
        _replace_attention_linear(
            attn,
            tp_group,
            column_names=["q_proj", "k_proj", "v_proj"],
            row_names=["o_proj"],
        )
        _patch_num_heads(attn, tp_size, ["num_heads", "num_key_value_heads"])
        _replace_mlp_linear(layer.mlp, tp_group)

    # 2. Cross-decoder layers
    for layer in text_model.cross_decoder.layers:
        attn = layer.cross_attn
        _replace_attention_linear(
            attn,
            tp_group,
            column_names=["q_proj", "k_proj", "v_proj"],
            row_names=["o_proj"],
        )
        _patch_num_heads(attn, tp_size, ["num_heads", "num_key_value_heads"])
        _replace_mlp_linear(layer.mlp, tp_group)

    # 3. Patch self-attention forward to use local head counts
    def self_attn_forward_tp(self, hidden_states, position_embeddings, cu_seqlens_q, max_seqlen_q,
                             past_key_value=None, use_cache=False):
        from flash_attn import flash_attn_varlen_func

        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        q = _apply_rotary_to_single(q, position_embeddings)
        k = _apply_rotary_to_single(k, position_embeddings)

        if past_key_value is None:
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=True,
                window_size=(self.window_size - 1, 0),
            )
            key_states = k
            value_states = v
        else:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, k], dim=0)
            value_states = torch.cat([past_v, v], dim=0)
            cu_seqlens_k = torch.tensor([0, key_states.shape[0]], dtype=torch.int32, device=hidden_states.device)
            attn_output = flash_attn_varlen_func(
                q,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=key_states.shape[0],
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=False,
            )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        next_past = None
        if use_cache:
            next_past = (
                key_states[-self.window_size :].contiguous(),
                value_states[-self.window_size :].contiguous(),
            )
        return self.o_proj(attn_output), next_past

    def cross_attn_forward_tp(self, hidden_states, memory_states, query_position_embeddings,
                              memory_position_embeddings, cu_seqlens_q, max_seqlen_q,
                              cu_seqlens_k, max_seqlen_k, causal):
        from flash_attn import flash_attn_varlen_func

        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(memory_states).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(memory_states).view(-1, self.num_key_value_heads, self.head_dim)

        q = _apply_rotary_to_single(q, query_position_embeddings)
        k = _apply_rotary_to_single(k, memory_position_embeddings)

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=causal,
        )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        return self.o_proj(attn_output)

    for layer in text_model.self_decoder.layers:
        layer.self_attn.forward = types.MethodType(self_attn_forward_tp, layer.self_attn)

    for layer in text_model.cross_decoder.layers:
        layer.cross_attn.forward = types.MethodType(cross_attn_forward_tp, layer.cross_attn)

    # 4. LM head
    lm_head = model.lm_head
    model.lm_head = ColumnParallelLinear.from_linear(lm_head, tp_group, gather_output=True)

