"""YOCO-specific data utilities.

This stage keeps the existing dataset preprocessing untouched and only defines
the packing logic needed by the YOCO baseline. The collate path uses ordinary
decoder inputs and ignores all Comb-specific chunk fields.
"""

import torch


def collate_fn_yoco(batch):
    """Pack variable-length decoder-only samples for YOCO.

    Expected per-sample fields:
    - ``input_ids``
    - ``shift_labels``

    Optional fields such as ``chunk_ids`` are ignored.
    """
    all_input_ids = []
    all_shift_labels = []
    all_position_ids = []
    cu_seqlens_q = [0]
    max_q = 0

    for item in batch:
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        shift_labels = torch.tensor(item["shift_labels"], dtype=torch.long)

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

