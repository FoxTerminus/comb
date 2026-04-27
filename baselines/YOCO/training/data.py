"""YOCO-specific data utilities for decoder-only packed batches."""

import torch


def _to_next_token_labels(input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Convert same-position supervised labels into causal next-token labels."""
    next_labels = torch.full_like(labels, -100)
    if labels.numel() > 1:
        supervised_next = labels[1:] != -100
        next_labels[:-1] = torch.where(supervised_next, input_ids[1:], next_labels[:-1])
    return next_labels


def collate_fn_yoco(
    batch,
    max_seq_len: int | None = None,
    label_shift_mode: str = "existing",
):
    """Pack variable-length YOCO samples into a decoder-only batch.

    Expected per-sample fields:
    - ``input_ids``
    - ``shift_labels``
    """
    all_input_ids = []
    all_shift_labels = []
    all_position_ids = []
    cu_seqlens_q = [0]
    max_q = 0

    for item in batch:
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        shift_labels = torch.tensor(item["shift_labels"], dtype=torch.long)

        if max_seq_len is not None and max_seq_len > 0 and input_ids.numel() > max_seq_len:
            input_ids = input_ids[-max_seq_len:]
            shift_labels = shift_labels[-max_seq_len:]

        if label_shift_mode == "next-token":
            shift_labels = _to_next_token_labels(input_ids, shift_labels)
        elif label_shift_mode != "existing":
            raise ValueError(f"Unsupported YOCO label_shift_mode: {label_shift_mode}")

        all_input_ids.append(input_ids)
        all_shift_labels.append(shift_labels)
        all_position_ids.append(torch.arange(len(input_ids), dtype=torch.long))

        cu_seqlens_q.append(cu_seqlens_q[-1] + len(input_ids))
        max_q = max(max_q, len(input_ids))

    return {
        "input_ids": torch.cat(all_input_ids).unsqueeze(0),
        "shift_labels": torch.cat(all_shift_labels).unsqueeze(0),
        "position_ids": torch.cat(all_position_ids).unsqueeze(0),
        "cu_seqlens_q": torch.tensor(cu_seqlens_q, dtype=torch.int32),
        "max_seqlen_q": max_q,
    }
