"""SambaY-specific training sample validation and collate utilities.

Raw dataset download and multi-turn chat tokenization live under
``baselines/SambaY/data``. This module receives those SambaY-native tokenized
examples and packs them into the model's decoder-only training interface.
"""

from __future__ import annotations

import torch


def _to_next_token_labels(input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Convert same-position supervised labels into causal next-token labels."""
    next_labels = torch.full_like(labels, -100)
    if labels.numel() > 1:
        supervised_next = labels[1:] != -100
        next_labels[:-1] = torch.where(supervised_next, input_ids[1:], next_labels[:-1])
    return next_labels


def preprocess_sambay_example(
    example: dict,
    max_seq_len: int | None = None,
    label_shift_mode: str = "existing",
) -> dict:
    """Validate and optionally truncate one SambaY-native tokenized item.

    The expected input is produced by ``baselines/SambaY/data`` from the raw
    multi-turn chat datasets. Comb fields such as ``chunk_ids`` are not part of
    the SambaY data contract and are ignored if a caller accidentally supplies
    them.
    """
    if "input_ids" not in example:
        raise KeyError("SambaY preprocessing requires input_ids.")
    if "shift_labels" not in example:
        raise KeyError("SambaY preprocessing requires shift_labels.")

    input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
    shift_labels = torch.tensor(example["shift_labels"], dtype=torch.long)
    if input_ids.shape != shift_labels.shape:
        raise ValueError(
            "input_ids and shift_labels must have the same shape before packing: "
            f"{tuple(input_ids.shape)} != {tuple(shift_labels.shape)}"
        )

    if max_seq_len is not None and max_seq_len > 0 and input_ids.numel() > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
        shift_labels = shift_labels[-max_seq_len:]

    if label_shift_mode == "next-token":
        shift_labels = _to_next_token_labels(input_ids, shift_labels)
    elif label_shift_mode != "existing":
        raise ValueError(f"Unsupported SambaY label_shift_mode: {label_shift_mode}")

    seq_len = int(input_ids.numel())
    return {
        "input_ids": input_ids,
        "shift_labels": shift_labels,
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "sequence_id": torch.zeros(seq_len, dtype=torch.long),
        "seq_len": seq_len,
    }


def collate_fn_sambay(
    batch,
    max_seq_len: int | None = None,
    label_shift_mode: str = "existing",
    include_position_ids: bool = False,
) -> dict:
    """Pack variable-length SambaY samples into one decoder stream."""
    all_input_ids = []
    all_shift_labels = []
    all_attention_mask = []
    all_sequence_ids = []
    cu_seqlens_q = [0]
    max_q = 0

    for sample_idx, item in enumerate(batch):
        processed = preprocess_sambay_example(
            item,
            max_seq_len=max_seq_len,
            label_shift_mode=label_shift_mode,
        )
        input_ids = processed["input_ids"]
        shift_labels = processed["shift_labels"]
        seq_len = processed["seq_len"]

        all_input_ids.append(input_ids)
        all_shift_labels.append(shift_labels)
        all_attention_mask.append(processed["attention_mask"])
        all_sequence_ids.append(torch.full((seq_len,), sample_idx, dtype=torch.long))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seq_len)
        max_q = max(max_q, seq_len)

    packed = {
        "input_ids": torch.cat(all_input_ids).unsqueeze(0),
        "shift_labels": torch.cat(all_shift_labels).unsqueeze(0),
        "attention_mask": torch.cat(all_attention_mask).unsqueeze(0),
        "sequence_ids": torch.cat(all_sequence_ids).unsqueeze(0),
        "cu_seqlens_q": torch.tensor(cu_seqlens_q, dtype=torch.int32),
        "max_seqlen_q": max_q,
    }
    if include_position_ids:
        position_ids = [torch.arange(end - start, dtype=torch.long) for start, end in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])]
        packed["position_ids"] = torch.cat(position_ids).unsqueeze(0)
    return packed
