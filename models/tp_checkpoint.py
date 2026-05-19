"""Checkpoint split/merge helpers for CombLlama tensor parallel shards."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Mapping

import torch


COLUMN_PARALLEL_PATTERNS = [
    r"chunk_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.weight",
    r"chunk_model\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight",
    r"chunk_model\.(k_proj|v_proj)\.\d+\.weight",
    r"language_model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.weight",
    r"language_model\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight",
    r"language_model\.model\.cross_layers\.\d+\.cross_attn\.q_proj\.weight",
    r"language_model\.model\.cross_layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight",
    r"language_model\.lm_head\.weight",
]

ROW_PARALLEL_PATTERNS = [
    r"chunk_model\.layers\.\d+\.self_attn\.o_proj\.weight",
    r"chunk_model\.layers\.\d+\.mlp\.down_proj\.weight",
    r"language_model\.model\.layers\.\d+\.self_attn\.o_proj\.weight",
    r"language_model\.model\.layers\.\d+\.mlp\.down_proj\.weight",
    r"language_model\.model\.cross_layers\.\d+\.cross_attn\.o_proj\.weight",
    r"language_model\.model\.cross_layers\.\d+\.mlp\.down_proj\.weight",
]


def is_column_parallel_key(key: str) -> bool:
    return any(re.fullmatch(pattern, key) for pattern in COLUMN_PARALLEL_PATTERNS)


def is_row_parallel_key(key: str) -> bool:
    return any(re.fullmatch(pattern, key) for pattern in ROW_PARALLEL_PATTERNS)


def split_state_dict_for_tp(state_dict: Mapping[str, torch.Tensor], tp_size: int) -> list[OrderedDict]:
    """Split a full CombLlama state dict into TP rank shards."""
    shards = [OrderedDict() for _ in range(tp_size)]
    for key, tensor in state_dict.items():
        if is_column_parallel_key(key):
            if tensor.shape[0] % tp_size != 0:
                raise ValueError(f"{key} dim0={tensor.shape[0]} must divide tp_size={tp_size}")
            chunks = torch.chunk(tensor, tp_size, dim=0)
        elif is_row_parallel_key(key):
            if tensor.shape[1] % tp_size != 0:
                raise ValueError(f"{key} dim1={tensor.shape[1]} must divide tp_size={tp_size}")
            chunks = torch.chunk(tensor, tp_size, dim=1)
        else:
            chunks = [tensor for _ in range(tp_size)]
        for rank in range(tp_size):
            shards[rank][key] = chunks[rank].clone()
    return shards


def merge_state_dict_from_tp(shards: list[Mapping[str, torch.Tensor]]) -> OrderedDict:
    """Merge TP rank shards back into a full CombLlama state dict."""
    if not shards:
        raise ValueError("`shards` must be non-empty.")
    merged = OrderedDict()
    keys = list(shards[0].keys())
    for key in keys:
        tensors = [shard[key] for shard in shards]
        if is_column_parallel_key(key):
            merged[key] = torch.cat(tensors, dim=0)
        elif is_row_parallel_key(key):
            merged[key] = torch.cat(tensors, dim=1)
        else:
            merged[key] = tensors[0].clone()
    return merged

