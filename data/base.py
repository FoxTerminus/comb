"""Packed data collation utilities for CombLlama training."""

from typing import Optional

import torch


DEFAULT_LABEL_KEY = "meta-llama/Llama-3.1-8B-Instruct"


def _as_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _build_shifted_labels(prompt_len: int, answer_ids: list[int]) -> list[int]:
    if prompt_len <= 0:
        raise ValueError("Each sample must contain at least one prompt token.")
    return [-100] * (prompt_len - 1) + answer_ids + [-100]


def collate_fn(batch, label_key: Optional[str] = DEFAULT_LABEL_KEY):
    """Pack variable-length CombLlama samples for FlashAttention varlen.

    Each sample is expected to contain:
    - ``input_ids``: prompt text tokens.
    - ``chunk_ids``: context tokens. Current model assumes one context chunk per
      sample, so this also defines the cross-attention K/V sequence.
    - ``label_key`` column: answer tokens. Defaults to
      ``meta-llama/Llama-3.1-8B-Instruct``. Other label columns must be
      requested explicitly.

    Returns tensors accepted by ``CombLlamaForConditionalGeneration.forward``.
    Labels are shifted here so ``logits[i]`` predicts ``labels[i]``.
    """
    all_input_ids = []
    all_chunk_ids = []
    all_labels = []
    all_position_ids = []
    all_position_ids_k = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0

    for item in batch:
        if label_key is None:
            raise ValueError("`label_key` must be specified. Use `labels` explicitly for original labels.")
        if label_key not in item:
            raise KeyError(f"`label_key={label_key}` is not present in the sample.")
        prompt_ids = _as_list(item["input_ids"])
        answer_ids = _as_list(item[label_key])
        chunk_ids = _as_list(item["chunk_ids"])

        if not chunk_ids:
            raise ValueError("Each sample must contain at least one context/chunk token.")

        text_ids = prompt_ids + answer_ids
        labels = _build_shifted_labels(len(prompt_ids), answer_ids)
        if len(labels) != len(text_ids):
            raise ValueError("Internal label construction error: labels and input_ids lengths differ.")

        all_input_ids.extend(text_ids)
        all_chunk_ids.extend(chunk_ids)
        all_labels.extend(labels)
        all_position_ids.extend(range(len(text_ids)))
        all_position_ids_k.extend(range(len(chunk_ids)))

        cu_seqlens_q.append(cu_seqlens_q[-1] + len(text_ids))
        cu_seqlens_k.append(cu_seqlens_k[-1] + len(chunk_ids))
        max_seqlen_q = max(max_seqlen_q, len(text_ids))
        max_seqlen_k = max(max_seqlen_k, len(chunk_ids))

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long).unsqueeze(0),
        "chunk_ids": torch.tensor(all_chunk_ids, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(all_position_ids, dtype=torch.long).unsqueeze(0),
        "position_ids_k": torch.tensor(all_position_ids_k, dtype=torch.long).unsqueeze(0),
        "cu_seqlens_q": torch.tensor(cu_seqlens_q, dtype=torch.int32),
        "max_seqlen_q": max_seqlen_q,
        "cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32),
        "max_seqlen_k": max_seqlen_k,
        "labels": torch.tensor(all_labels, dtype=torch.long).unsqueeze(0),
    }
