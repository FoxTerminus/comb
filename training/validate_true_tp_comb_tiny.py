#!/usr/bin/env python3
"""Numerically compare a tiny unsharded CombLlama with the experimental TP path."""

from __future__ import annotations

import copy
import json
import os

import deepspeed
import torch
import torch.distributed as dist
from transformers import LlamaConfig

from comb.integration.hf.CombLlama import (
    CombLlamaConfig,
    CombLlamaForConditionalGeneration,
)
from training.true_tp_comb_adapter import apply_true_tp_comb_adapter


def make_model(num_key_value_heads: int = 2) -> CombLlamaForConditionalGeneration:
    text_config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    config = CombLlamaConfig(
        text_config=text_config,
        chunk_token_index=127,
        num_hidden_layers=3,
        cross_attention_layers=[1],
        pad_token_id=0,
    )
    model = CombLlamaForConditionalGeneration(config=config, from_scratch=False)
    for parameter in model.language_model.parameters():
        parameter.requires_grad = False
    for parameter in model.chunk_model.embed_tokens.parameters():
        parameter.requires_grad = False
    for parameter in model.language_model.model.cross_layers.parameters():
        parameter.requires_grad = True
    return model


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42)

    # Keep the default fixture small for TP=1/2 checkpoint tests, but exercise
    # a production-valid KV-head layout when this test is launched with TP=4.
    full = make_model(num_key_value_heads=max(2, world_size)).to(
        local_rank, dtype=torch.float32
    ).train()
    no_op_probe = copy.deepcopy(full)
    no_op_module_ids = {name: id(module) for name, module in no_op_probe.named_modules()}
    assert apply_true_tp_comb_adapter(no_op_probe, tp_size=1, dtype=torch.float32) is no_op_probe
    assert no_op_module_ids == {
        name: id(module) for name, module in no_op_probe.named_modules()
    }
    assert not hasattr(no_op_probe, "_comb_true_tp")
    sharded = copy.deepcopy(full)
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=1e-3, weight_decay=0.01)

    input_ids = torch.randint(3, 120, (2, 8), device=local_rank)
    chunk_ids = torch.randint(3, 120, (2, 8), device=local_rank)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(chunk_ids, src=0)
    attention_mask = torch.ones_like(input_ids)
    chunk_mask = torch.ones_like(chunk_ids)

    full_output = full(
        input_ids=input_ids,
        chunk_ids=chunk_ids,
        attention_mask=attention_mask,
        cross_attention_mask=chunk_mask,
        use_cache=False,
    )
    full_loss = full_output.logits.float().square().mean()
    full_loss.backward()

    apply_true_tp_comb_adapter(sharded, tp_size=world_size, dtype=torch.float32)
    sharded.to(local_rank, dtype=torch.float32).train()
    tp_optimizer = torch.optim.AdamW(sharded.parameters(), lr=1e-3, weight_decay=0.01)
    pic_projection_shapes = []
    pic_hook = sharded.chunk_model.k_proj[0].register_forward_hook(
        lambda _module, _inputs, output: pic_projection_shapes.append(list(output.shape))
    )
    tp_output = sharded(
        input_ids=input_ids,
        chunk_ids=chunk_ids,
        attention_mask=attention_mask,
        cross_attention_mask=chunk_mask,
        use_cache=False,
    )
    pic_hook.remove()
    tp_loss = tp_output.logits.float().square().mean()
    tp_loss.backward()

    full_named = dict(full.named_parameters())
    tp_named = dict(sharded.named_parameters())
    shared_names = (
        "language_model.model.cross_layers.0.cross_attn.q_norm.weight",
        "language_model.model.cross_layers.0.cross_attn.k_norm.weight",
        "language_model.model.cross_layers.0.cross_attn_attn_gate",
        "language_model.model.cross_layers.0.cross_attn_mlp_gate",
        "language_model.model.cross_layers.0.input_layernorm.weight",
    )
    shared_gradient_deltas = {}
    for name in shared_names:
        expected = full_named[name].grad
        actual = tp_named[name].grad
        shared_gradient_deltas[name] = (
            None
            if expected is None or actual is None
            else float((expected - actual).abs().max())
        )

    # The added chunk K/V projections are column-sharded along output rows.
    column_gradient_deltas = {}
    for name in (
        "chunk_model.k_proj.0.weight",
        "chunk_model.v_proj.0.weight",
        "language_model.model.cross_layers.0.cross_attn.q_proj.weight",
    ):
        expected_shard = torch.chunk(full_named[name].grad, world_size, dim=0)[rank]
        column_gradient_deltas[name] = float(
            (expected_shard - tp_named[name].grad).abs().max()
        )

    full_optimizer.step()
    tp_optimizer.step()
    parameter_deltas = {}
    for name, actual in sharded.named_parameters():
        if not actual.requires_grad:
            continue
        expected = full_named[name]
        if expected.shape == actual.shape:
            expected_local = expected
        else:
            shard_dims = [
                dim
                for dim, (full_dim, local_dim) in enumerate(
                    zip(expected.shape, actual.shape)
                )
                if full_dim == local_dim * world_size
                and all(
                    expected.shape[other] == actual.shape[other]
                    for other in range(expected.ndim)
                    if other != dim
                )
            ]
            assert len(shard_dims) == 1, (name, expected.shape, actual.shape)
            expected_local = torch.chunk(expected, world_size, dim=shard_dims[0])[rank]
        parameter_deltas[name] = float(
            (expected_local.detach() - actual.detach()).abs().max()
        )

    report = {
        "rank": rank,
        "tp_size": world_size,
        "full_loss": float(full_loss.detach()),
        "tp_loss": float(tp_loss.detach()),
        "logit_max_abs_delta": float(
            (full_output.logits - tp_output.logits).detach().abs().max()
        ),
        "local_k_projection_shape": pic_projection_shapes[0],
        "adapter": sharded._comb_true_tp,
        "shared_gradient_max_abs_delta": shared_gradient_deltas,
        "column_gradient_max_abs_delta": column_gradient_deltas,
        "all_trainable_parameter_max_abs_delta_after_adamw": max(
            parameter_deltas.values()
        ),
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, report)
    if rank == 0:
        print(json.dumps(gathered, indent=2), flush=True)
    assert abs(float(full_loss.detach()) - float(tp_loss.detach())) < 1e-7
    assert report["logit_max_abs_delta"] < 1e-5
    assert all(
        delta is None or delta < 1e-7
        for delta in shared_gradient_deltas.values()
    )
    assert all(delta < 1e-7 for delta in column_gradient_deltas.values())
    assert max(parameter_deltas.values()) < 1e-7
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
