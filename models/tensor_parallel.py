"""Tensor-parallel adapters for the packed-FA CombLlama model.

This is a small Megatron-style TP layer set for ``comb.models.CombLlama``.
It deliberately does not depend on DeepSpeed AutoTP: every sharded projection is
chosen explicitly so packed FlashAttention and cross-attention keep their
intended semantics.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Function
from transformers.modeling_outputs import CausalLMOutput


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _AllGatherLastDim(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        ctx.tp_size = dist.get_world_size(group)
        ctx.tp_rank = dist.get_rank(group)
        gathered = [torch.empty_like(tensor) for _ in range(ctx.tp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        chunks = torch.chunk(grad_output, ctx.tp_size, dim=-1)
        return chunks[ctx.tp_rank].contiguous(), None


class _CopyToTPRegion(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.group is None or dist.get_world_size(ctx.group) == 1:
            return grad_output, None
        grad_input = grad_output.contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


def all_reduce(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return _AllReduce.apply(tensor, group)


def all_gather_last_dim(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return _AllGatherLastDim.apply(tensor, group)


def copy_to_tensor_parallel_region(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return _CopyToTPRegion.apply(tensor, group)


class _VocabParallelCrossEntropy(Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.input_shape = local_logits.shape

        probs = local_logits.reshape(-1, local_logits.shape[-1]).float()
        labels_1d = labels.reshape(-1)
        valid = labels_1d.ne(-100)

        local_max = probs.max(dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

        local_target_index = (labels_1d - vocab_start_index).clamp(min=0, max=probs.shape[-1] - 1)
        target_in_rank = valid & labels_1d.ge(vocab_start_index) & labels_1d.lt(vocab_end_index)
        local_target_logits = probs.gather(1, local_target_index.unsqueeze(1)).squeeze(1)
        local_target_logits = torch.where(target_in_rank, local_target_logits, torch.zeros_like(local_target_logits))
        target_logits = local_target_logits.clone()
        dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=group)

        probs.sub_(global_max.unsqueeze(-1)).exp_()
        sum_exp = probs.sum(dim=-1)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        losses = sum_exp.log() + global_max - target_logits
        losses = torch.where(valid, losses, torch.zeros_like(losses))
        denom = valid.sum().clamp_min(1)

        probs.div_(sum_exp.unsqueeze(-1))
        ctx.save_for_backward(probs, local_target_index, target_in_rank, valid, denom)
        return losses.sum() / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, local_target_index, target_in_rank, valid, denom = ctx.saved_tensors
        grad = softmax
        if target_in_rank.any():
            rows = torch.arange(grad.shape[0], device=grad.device)
            grad[rows[target_in_rank], local_target_index[target_in_rank]] -= 1.0
        grad = grad * valid.unsqueeze(1).to(grad.dtype)
        grad = grad / denom.to(grad.dtype)
        grad = grad.reshape(ctx.input_shape)
        grad = grad * grad_output.to(grad.dtype)
        return grad, None, None, None, None


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    return _VocabParallelCrossEntropy.apply(local_logits, labels, vocab_start_index, vocab_end_index, group)


class ColumnParallelLinear(nn.Module):
    """Linear with output features split across TP ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        if out_features % self.tp_size != 0:
            raise ValueError(f"out_features={out_features} must divide tp_size={self.tp_size}")
        self.out_features_per_rank = out_features // self.tp_size
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
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            tp_group=tp_group,
            gather_output=gather_output,
        )
        layer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        chunk = linear.out_features // layer.tp_size
        start = layer.tp_rank * chunk
        end = start + chunk
        with torch.no_grad():
            layer.weight.copy_(linear.weight[start:end])
            if linear.bias is not None:
                layer.bias.copy_(linear.bias[start:end])
        layer.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            layer.bias.requires_grad_(linear.bias.requires_grad)
        return layer


class RowParallelLinear(nn.Module):
    """Linear with input features split across TP ranks and output all-reduced."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.in_features = in_features
        self.out_features = out_features
        if in_features % self.tp_size != 0:
            raise ValueError(f"in_features={in_features} must divide tp_size={self.tp_size}")
        self.in_features_per_rank = in_features // self.tp_size
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
    def from_linear(cls, linear: nn.Linear, tp_group: dist.ProcessGroup) -> "RowParallelLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            tp_group=tp_group,
        )
        layer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        chunk = linear.in_features // layer.tp_size
        start = layer.tp_rank * chunk
        end = start + chunk
        with torch.no_grad():
            layer.weight.copy_(linear.weight[:, start:end])
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        layer.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            layer.bias.requires_grad_(linear.bias.requires_grad)
        return layer


def _replace_attention(attn_module: nn.Module, tp_group: dist.ProcessGroup, column_names, row_names) -> None:
    for name in column_names:
        setattr(attn_module, name, ColumnParallelLinear.from_linear(getattr(attn_module, name), tp_group))
    for name in row_names:
        setattr(attn_module, name, RowParallelLinear.from_linear(getattr(attn_module, name), tp_group))


def _replace_mlp(mlp_module: nn.Module, tp_group: dist.ProcessGroup) -> None:
    mlp_module.gate_proj = ColumnParallelLinear.from_linear(mlp_module.gate_proj, tp_group)
    mlp_module.up_proj = ColumnParallelLinear.from_linear(mlp_module.up_proj, tp_group)
    mlp_module.down_proj = RowParallelLinear.from_linear(mlp_module.down_proj, tp_group)


def _patch_head_attrs(module: nn.Module, tp_size: int, attrs) -> None:
    module._tp_size = tp_size
    for attr in attrs:
        if hasattr(module, attr):
            value = getattr(module, attr)
            if value % tp_size != 0:
                raise ValueError(f"{module.__class__.__name__}.{attr}={value} must divide tp_size={tp_size}")
            setattr(module, attr, value // tp_size)


def _patch_text_self_attention_forward(text_model: nn.Module, tp_size: int) -> None:
    text_model._tp_size = tp_size
    text_model._local_num_heads = text_model.text_config.num_attention_heads // tp_size
    text_model._local_num_key_value_heads = text_model.text_config.num_key_value_heads // tp_size

    def _self_attn_forward_tp(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        position_embeddings,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        num_heads = self._local_num_heads
        num_key_value_heads = self._local_num_key_value_heads
        head_dim = decoder_layer.self_attn.head_dim

        q = decoder_layer.self_attn.q_proj(hidden_states).view(-1, num_heads, head_dim)
        k = decoder_layer.self_attn.k_proj(hidden_states).view(-1, num_key_value_heads, head_dim)
        v = decoder_layer.self_attn.v_proj(hidden_states).view(-1, num_key_value_heads, head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q, k = q.squeeze(0), k.squeeze(0)

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            dropout_p=0.0 if not self.training else self.text_config.attention_dropout,
            softmax_scale=decoder_layer.self_attn.scaling,
            causal=True,
        )
        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        hidden_states = residual + decoder_layer.self_attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        return residual + hidden_states

    text_model._self_attn_forward = types.MethodType(_self_attn_forward_tp, text_model)


def _patch_language_model_forward(language_model: nn.Module, tp_group: dist.ProcessGroup) -> None:
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    vocab_per_rank = language_model.vocab_size // tp_size
    vocab_start_index = tp_rank * vocab_per_rank
    vocab_end_index = vocab_start_index + vocab_per_rank

    language_model._tp_group = tp_group
    language_model._tp_size = tp_size
    language_model._tp_rank = tp_rank
    language_model._vocab_start_index = vocab_start_index
    language_model._vocab_end_index = vocab_end_index
    language_model._tp_gather_output_logits = False

    def _forward_tp(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        cross_attention_states=None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        output_hidden_states: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        loss_labels = labels
        if isinstance(logits_to_keep, torch.Tensor):
            hidden_states = hidden_states[:, logits_to_keep, :]
            if labels is not None:
                loss_labels = labels[:, logits_to_keep]
        elif logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
            if labels is not None:
                loss_labels = labels[:, -logits_to_keep:]

        local_logits = self.lm_head(hidden_states)
        loss = vocab_parallel_cross_entropy(
            local_logits,
            loss_labels,
            self._tp_group,
            self._vocab_start_index,
            self._vocab_end_index,
        )
        logits = local_logits
        if self._tp_gather_output_logits:
            logits = all_gather_last_dim(local_logits, self._tp_group)

        return CausalLMOutput(
            loss=loss,
            logits=logits.float(),
            hidden_states=outputs.hidden_states,
        )

    language_model.forward = types.MethodType(_forward_tp, language_model)


def apply_tensor_parallelism(model: nn.Module, tp_group: dist.ProcessGroup) -> nn.Module:
    """Shard the current packed-FA CombLlama model across ``tp_group``.

    The function mutates ``model`` in-place and returns it.
    """
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return model

    chunk_model = model.chunk_model
    for layer in chunk_model.layers:
        attn = layer.self_attn
        _replace_attention(attn, tp_group, ["q_proj", "k_proj", "v_proj"], ["o_proj"])
        _patch_head_attrs(attn, tp_size, ["num_heads", "num_key_value_heads"])
        _replace_mlp(layer.mlp, tp_group)

    for idx in range(len(chunk_model.k_proj)):
        chunk_model.k_proj[idx] = ColumnParallelLinear.from_linear(chunk_model.k_proj[idx], tp_group)
        chunk_model.v_proj[idx] = ColumnParallelLinear.from_linear(chunk_model.v_proj[idx], tp_group)
    _patch_head_attrs(chunk_model, tp_size, ["num_key_value_heads"])

    text_model = model.language_model.model
    for decoder_layer in text_model.layers:
        attn = decoder_layer.self_attn
        _replace_attention(attn, tp_group, ["q_proj", "k_proj", "v_proj"], ["o_proj"])
        _patch_head_attrs(attn, tp_size, ["num_heads", "num_key_value_heads"])
        _replace_mlp(decoder_layer.mlp, tp_group)
    _patch_text_self_attention_forward(text_model, tp_size)

    for cross_layer in text_model.cross_layers:
        cross_attn = cross_layer.cross_attn
        _replace_attention(cross_attn, tp_group, ["q_proj"], ["o_proj"])
        _patch_head_attrs(cross_attn, tp_size, ["num_heads"])
        _replace_mlp(cross_layer.mlp, tp_group)

    model.language_model.lm_head = ColumnParallelLinear.from_linear(
        model.language_model.lm_head,
        tp_group,
        gather_output=False,
    )
    _patch_language_model_forward(model.language_model, tp_group)
    model._tp_size = tp_size
    return model
