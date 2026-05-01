"""Megatron Tensor Parallel adaptation for CombLlama.

Provides functions to replace nn.Linear layers in a CombLlama model with
tensor-parallel (TP) equivalents, enabling DP+TP training on multi-GPU setups.

Usage:
    from models.CombLlama_megatron import apply_tensor_parallelism
    model = CombLlamaForConditionalGeneration(...)
    apply_tensor_parallelism(model, tp_group)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional


# ---------------------------------------------------------------------------
# Autograd-safe collective operations
# ---------------------------------------------------------------------------

class _AllReduceFunc(Function):
    """All-reduce that preserves the autograd graph.

    Forward:  output = all_reduce_sum(input)
    Backward: grad flows through unchanged (each rank already holds the full gradient).
    """

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _AllGatherFunc(Function):
    """All-gather along the last dimension that preserves the autograd graph.

    Forward:  output = cat(all_gather(input), dim=-1)
    Backward: split grad and take local rank's shard.
    """

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        ctx.tp_size = dist.get_world_size(group)
        ctx.tp_rank = dist.get_rank(group)
        tensors = [torch.empty_like(tensor) for _ in range(ctx.tp_size)]
        dist.all_gather(tensors, tensor, group=group)
        return torch.cat(tensors, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        chunks = torch.chunk(grad_output, ctx.tp_size, dim=-1)
        return chunks[ctx.tp_rank].contiguous(), None


class _CopyToTensorParallelRegionFunc(Function):
    """Identity in forward, all-reduce gradients in backward.

    Column-parallel layers consume replicated inputs and produce sharded outputs.
    Their input gradients are partial on each TP rank and must be summed before
    flowing into previous replicated activations.
    """

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is None or dist.get_world_size(ctx.group) == 1:
            return grad_output, None
        grad_input = grad_output.contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


def all_reduce(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Autograd-safe all-reduce sum."""
    return _AllReduceFunc.apply(tensor, group)


def all_gather_last_dim(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Autograd-safe all-gather along the last dimension."""
    return _AllGatherFunc.apply(tensor, group)


def copy_to_tensor_parallel_region(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Replicate input in forward and sum its gradient across TP ranks."""
    return _CopyToTensorParallelRegionFunc.apply(tensor, group)


# ---------------------------------------------------------------------------
# TP Linear Layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer with weight split along the output dimension across TP ranks.

    Each rank holds weight of shape [out_features // tp_size, in_features].
    Output is the local shard (not gathered) — use gather_output=True to
    all-gather the full output when needed (e.g., lm_head for loss computation).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group else 1
        self.gather_output = gather_output

        assert out_features % self.tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({self.tp_size})"
        )
        self.out_features_per_rank = out_features // self.tp_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            x = copy_to_tensor_parallel_region(x, self.tp_group)
        output = nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            output = all_gather_last_dim(output, self.tp_group)
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        tp_group: dist.ProcessGroup,
        gather_output: bool = False,
    ) -> "ColumnParallelLinear":
        """Create from an existing nn.Linear, splitting weight along dim=0."""
        tp_rank = dist.get_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)
        out_features = linear.out_features
        chunk_size = out_features // tp_size

        layer = cls(
            linear.in_features, out_features,
            bias=linear.bias is not None,
            tp_group=tp_group,
            gather_output=gather_output,
        )
        with torch.no_grad():
            layer.weight.copy_(
                linear.weight.data[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
            )
            if linear.bias is not None:
                layer.bias.copy_(
                    linear.bias.data[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
                )
        return layer


class RowParallelLinear(nn.Module):
    """Linear layer with weight split along the input dimension across TP ranks.

    Each rank holds weight of shape [out_features, in_features // tp_size].
    The output is all-reduced across TP ranks to produce the correct result.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group else 1

        assert in_features % self.tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({self.tp_size})"
        )
        self.in_features_per_rank = in_features // self.tp_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            output = all_reduce(output, self.tp_group)
        if self.bias is not None:
            output = output + self.bias
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        tp_group: dist.ProcessGroup,
    ) -> "RowParallelLinear":
        """Create from an existing nn.Linear, splitting weight along dim=1."""
        tp_rank = dist.get_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)
        in_features = linear.in_features
        chunk_size = in_features // tp_size

        layer = cls(
            in_features, linear.out_features,
            bias=linear.bias is not None,
            tp_group=tp_group,
        )
        with torch.no_grad():
            layer.weight.copy_(
                linear.weight.data[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
            )
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.data)
        return layer


# ---------------------------------------------------------------------------
# TP Adaptation for CombLlama
# ---------------------------------------------------------------------------

def _replace_attention_linear(attn_module, tp_group, column_names, row_names):
    """Replace linear layers in an attention module with TP versions."""
    for name in column_names:
        old = getattr(attn_module, name)
        setattr(attn_module, name, ColumnParallelLinear.from_linear(old, tp_group))
    for name in row_names:
        old = getattr(attn_module, name)
        setattr(attn_module, name, RowParallelLinear.from_linear(old, tp_group))


def _replace_mlp_linear(mlp_module, tp_group):
    """Replace linear layers in a LlamaMLP with TP versions."""
    mlp_module.gate_proj = ColumnParallelLinear.from_linear(mlp_module.gate_proj, tp_group)
    mlp_module.up_proj = ColumnParallelLinear.from_linear(mlp_module.up_proj, tp_group)
    mlp_module.down_proj = RowParallelLinear.from_linear(mlp_module.down_proj, tp_group)


def _patch_num_heads(module, tp_size, head_attrs):
    """Store tp_size and adjust head count attributes on a module."""
    module._tp_size = tp_size
    for attr in head_attrs:
        if hasattr(module, attr):
            old_val = getattr(module, attr)
            setattr(module, attr, old_val // tp_size)


def apply_tensor_parallelism(model, tp_group: dist.ProcessGroup):
    """Apply tensor parallelism to a CombLlamaForConditionalGeneration model.

    Replaces nn.Linear layers with TP-sharded versions and adjusts attention
    head counts for local computation. The model must already have weights loaded.

    Args:
        model: A CombLlamaForConditionalGeneration instance.
        tp_group: The torch.distributed process group for tensor parallelism.
    """
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return

    # --- 1. Chunk Encoder (model.chunk_model) ---
    chunk_model = model.chunk_model

    # 1a. ChunkVarlenLayer self-attention + MLP
    for layer in chunk_model.layers:
        attn = layer.self_attn
        _replace_attention_linear(
            attn, tp_group,
            column_names=["q_proj", "k_proj", "v_proj"],
            row_names=["o_proj"],
        )
        _patch_num_heads(attn, tp_size, ["num_heads", "num_kv_heads"])
        _replace_mlp_linear(layer.mlp, tp_group)

    # 1b. Chunk encoder k_proj / v_proj (ModuleList of nn.Linear)
    tp_rank = dist.get_rank(tp_group)
    for idx in range(len(chunk_model.k_proj)):
        chunk_model.k_proj[idx] = ColumnParallelLinear.from_linear(
            chunk_model.k_proj[idx], tp_group
        )
        chunk_model.v_proj[idx] = ColumnParallelLinear.from_linear(
            chunk_model.v_proj[idx], tp_group
        )

    # Patch chunk_model head counts for view operations
    chunk_model.num_key_value_heads = chunk_model.num_key_value_heads // tp_size

    # --- 2. Text Decoder (model.language_model.model) ---
    text_model = model.language_model.model

    # 2a. Self-attention layers (LlamaDecoderLayer)
    for decoder_layer in text_model.layers:
        attn = decoder_layer.self_attn
        _replace_attention_linear(
            attn, tp_group,
            column_names=["q_proj", "k_proj", "v_proj"],
            row_names=["o_proj"],
        )
        _patch_num_heads(attn, tp_size, ["num_heads", "num_key_value_heads"])
        _replace_mlp_linear(decoder_layer.mlp, tp_group)

    # Patch text_model config head counts for _self_attn_forward
    # Store original values and local values
    text_model._tp_size = tp_size
    text_model._local_num_heads = text_model.config.num_attention_heads // tp_size
    text_model._local_num_kv_heads = text_model.config.num_key_value_heads // tp_size

    # Monkey-patch _self_attn_forward to use local head counts
    import types

    def _self_attn_forward_tp(self, decoder_layer, hidden_states, position_embeddings,
                              cu_seqlens_q, max_seqlen_q):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        from flash_attn import flash_attn_varlen_func

        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        num_heads = self._local_num_heads
        num_kv_heads = self._local_num_kv_heads
        head_dim = decoder_layer.self_attn.head_dim

        q = decoder_layer.self_attn.q_proj(hidden_states).view(-1, num_heads, head_dim)
        k = decoder_layer.self_attn.k_proj(hidden_states).view(-1, num_kv_heads, head_dim)
        v = decoder_layer.self_attn.v_proj(hidden_states).view(-1, num_kv_heads, head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)

        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            softmax_scale=decoder_layer.self_attn.scaling,
            causal=True,
        )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        hidden_states = residual + decoder_layer.self_attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    text_model._self_attn_forward = types.MethodType(_self_attn_forward_tp, text_model)

    # 2b. Cross-attention layers
    for cross_layer in text_model.cross_layers:
        cross_attn = cross_layer.cross_attn
        _replace_attention_linear(
            cross_attn, tp_group,
            column_names=["q_proj"],
            row_names=["o_proj"],
        )
        _patch_num_heads(cross_attn, tp_size, ["num_heads"])
        _replace_mlp_linear(cross_layer.mlp, tp_group)

    # --- 3. LM Head ---
    lm_head = model.language_model.lm_head
    model.language_model.lm_head = ColumnParallelLinear.from_linear(
        lm_head, tp_group, gather_output=True,
    )
