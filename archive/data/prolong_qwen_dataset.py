"""ProLong-64K Qwen3-tokenized packed pretraining dataset.

Reuses the LITPKDS binary format from ``baselines/ArchScale/data/prolong_dataset.py``
adapted for Qwen3 tokenizer (vocab_size=151936).
Returns ``{"input_ids": ..., "labels": ...}``; chunk/collate handled by training script.
"""

from __future__ import annotations

import glob
import os
import random
import struct
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# LITPKDS header (identical to ArchScale version)
# ---------------------------------------------------------------------------

HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24

DTYPE_MAP: dict[int, np.dtype] = {
    1: np.dtype(np.uint8), 2: np.dtype(np.int8), 3: np.dtype(np.int16),
    4: np.dtype(np.int32), 5: np.dtype(np.int64), 6: np.dtype(np.float32),
    7: np.dtype(np.float64), 8: np.dtype(np.uint16),
}


def _read_header(path: str) -> tuple[np.dtype, int]:
    with open(path, "rb") as f:
        magic = f.read(len(HDR_MAGIC))
        if magic != HDR_MAGIC:
            raise ValueError(f"Bad LITPKDS magic in {path}: {magic!r}")
        (version,) = struct.unpack("<Q", f.read(8))
        if version != 1:
            raise ValueError(f"Unsupported LITPKDS version {version}")
        (dtype_code,) = struct.unpack("<B", f.read(1))
        if dtype_code not in DTYPE_MAP:
            raise ValueError(f"Unknown dtype code {dtype_code}")
        (packed_chunk_size,) = struct.unpack("<Q", f.read(8))
    return DTYPE_MAP[dtype_code], packed_chunk_size


def _file_num_tokens(path: str, dtype: np.dtype) -> int:
    payload_bytes = os.path.getsize(path) - HDR_SIZE
    if payload_bytes % dtype.itemsize != 0:
        raise ValueError(
            f"Payload {payload_bytes} bytes not divisible by {dtype.itemsize}"
        )
    return payload_bytes // dtype.itemsize


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProLongQwenDataset(Dataset):
    """Map-style dataset over Qwen3-tokenized ProLong-64K .bin files.

    Each .bin contains continuous token IDs. Blocks are pre-scanned and
    shuffled at init time. Chunk/collate logic is handled by the training script.

    Data contract:
    - Returns ``{"input_ids": ..., "labels": ...}``
    - ``labels == input_ids`` (model does no shift; training script/collate
      must create shift_labels from labels before calling model)
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int = 65536,
        seed: int = 42,
        stride: int | None = None,
    ):
        super().__init__()
        self.block_size = block_size
        stride = stride or block_size  # default: non-overlapping

        self.filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not self.filenames:
            raise FileNotFoundError(f"No .bin files in {data_dir}")

        self._blocks: list[tuple[str, int, int, np.dtype]] = []
        total_tokens = 0

        for path in self.filenames:
            dtype, _ = _read_header(path)
            num_tokens = _file_num_tokens(path, dtype)
            # Sliding windows: each sample = (start, length) with stride
            start = 0
            while start + block_size <= num_tokens:
                self._blocks.append((path, start, block_size, dtype))
                start += stride
            total_tokens += num_tokens

        rng = random.Random(seed)
        rng.shuffle(self._blocks)

        self._total_tokens = total_tokens
        self._total_blocks = len(self._blocks)

    def __len__(self) -> int:
        return self._total_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, start, length, dtype = self._blocks[idx]
        arr = np.memmap(path, dtype=dtype, mode="r", offset=HDR_SIZE)
        tokens = torch.from_numpy(
            arr[start : start + length].astype(np.int64).copy()
        ).long()
        return {"input_ids": tokens, "labels": tokens.clone()}

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    def summary(self) -> str:
        return (
            f"ProLongQwenDataset: {len(self.filenames)} files, "
            f"{self._total_tokens / 1e9:.2f}B tokens, "
            f"{self._total_blocks} blocks of {self.block_size}"
        )
