"""DatasetBase: abstract base class for CombLlama training datasets.

Provides tokenizer initialization, dataset caching, and a collate function
that packs variable-length samples into flash-attention-compatible tensors.
"""

import os
import torch
from abc import ABC, abstractmethod
from datasets import load_from_disk
from transformers import AutoTokenizer

CPU_NUM = os.cpu_count()
HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')


def collate_fn(batch):
    """Collate variable-length samples into packed tensors for CombLlama.

    Packs multiple samples by concatenating all sequences (no padding) and
    tracking boundaries via cumulative sequence length tensors (cu_seqlens),
    as required by ``flash_attn_varlen_func``.

    Three sets of cu_seqlens are produced:
      - ``cu_seqlens_q``: per-sample decoder input boundaries.
      - ``cu_seqlens_k``: per-sample total chunk length boundaries (for cross-attention).
      - ``cu_seqlens_chunk``: per-chunk boundaries (for chunk self-attention).

    Note:
        Each sample must have at least one non-empty chunk in ``chunk_ids``.
        Empty chunks are silently skipped. If ALL chunks in the batch are empty,
        an assertion error is raised (flash attention requires ``max_seqlen > 0``).

    Args:
        batch: List of samples, each with ``input_ids``, ``shift_labels``,
            and ``chunk_ids`` (list of token id lists).

    Returns:
        Dict with packed tensors and metadata for the model's forward pass.
    """
    all_input_ids, all_chunk_ids, all_shift_labels = [], [], []
    all_pos_q, all_pos_k = [], []
    cu_seqlens_q = [0]      # Per-sample decoder boundaries
    cu_seqlens_k = [0]      # Per-sample total chunk length (for cross-attention)
    cu_seqlens_chunk = [0]  # Per-chunk boundaries (for chunk self-attention)
    max_q, max_k, max_chunk = 0, 0, 0

    for item in batch:
        # 1. Decoder inputs
        q_ids = torch.tensor(item['input_ids'])
        labels = torch.tensor(item['shift_labels'])
        all_input_ids.append(q_ids)
        all_shift_labels.append(labels)
        all_pos_q.append(torch.arange(len(q_ids)))
        cu_seqlens_q.append(cu_seqlens_q[-1] + len(q_ids))
        max_q = max(max_q, len(q_ids))

        # 2. Chunk encoder inputs
        sample_chunks_len = 0
        for chunk in item['chunk_ids']:
            if not chunk:  # Skip empty chunks to prevent max_seqlen=0
                continue
            c_ids = torch.tensor(chunk)
            all_chunk_ids.append(c_ids)
            all_pos_k.append(torch.arange(len(c_ids)))
            chunk_len = len(c_ids)
            sample_chunks_len += chunk_len
            cu_seqlens_chunk.append(cu_seqlens_chunk[-1] + chunk_len)
            max_chunk = max(max_chunk, chunk_len)

        cu_seqlens_k.append(cu_seqlens_k[-1] + sample_chunks_len)
        max_k = max(max_k, sample_chunks_len)

    assert all_chunk_ids, (
        "No non-empty chunks found in batch. Each sample must have at least one "
        "non-empty chunk — empty chunks cause flash_attn_varlen_func to crash "
        "with CUDA error: invalid configuration argument (max_seqlen=0)."
    )

    return {
        "input_ids": torch.cat(all_input_ids).unsqueeze(0),          # [1, total_input_len]
        "chunk_ids": torch.cat(all_chunk_ids).unsqueeze(0),          # [1, total_chunk_len]
        "shift_labels": torch.cat(all_shift_labels).unsqueeze(0),    # [1, total_input_len]
        "cu_seqlens_q": torch.tensor(cu_seqlens_q, dtype=torch.int32),
        "cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32),
        "cu_seqlens_chunk": torch.tensor(cu_seqlens_chunk, dtype=torch.int32),
        "max_seqlen_q": max_q,
        "max_seqlen_k": max_k,
        "max_seqlen_chunk": max_chunk,
        "position_ids": torch.cat(all_pos_q).unsqueeze(0),           # [1, total_input_len]
        "position_ids_k": torch.cat(all_pos_k).unsqueeze(0),         # [1, total_chunk_len]
    }


class DatasetBase(ABC):
    """Abstract base class for CombLlama training datasets.

    Handles tokenizer initialization, dataset caching, and provides
    a standard interface for dataset loading and access.

    Subclasses must implement :meth:`_init_data` to define dataset-specific
    loading and preprocessing logic.

    Args:
        model_name: HuggingFace model name for tokenizer initialization.
        split: Dataset split to load (e.g., ``"train"``, ``"train_sft"``).
        max_input_length: Maximum input sequence length (reserved for subclass use).
    """

    def __init__(self, model_name, split="train", max_input_length=512):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self._init_tokenizer()
        cached_path = HF_HOME + f'/datasets/{self.name}_{model_name.replace('/', '_')}'
        if split == "train" and os.path.exists(cached_path):
            self.data = load_from_disk(cached_path)
        else:
            self._init_data(split)

    @abstractmethod
    def _init_data(self, split):
        raise NotImplementedError

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<PAD>'
            self.tokenizer.pad_token_id = 128004

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
