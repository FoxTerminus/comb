"""ProLong-64k packed pretraining dataset.

Reads pre-tokenized LITPKDS ``.bin`` files from
``jsun/Prolong_64K_v2_Llama2_Tokenizer``.

Each ``.bin`` file has a 24-byte header::

    Offset  Size  Field
    ------  ----  --------------------
    0       7     Magic ``b"LITPKDS"``
    7       8     Version (uint64 LE)
    15      1     Dtype code
    16      8     Chunk size (element count, uint64 LE)

Dtype codes::  {1: uint8, 2: int8, 3: int16, 4: int32, 5: int64, 6: float32,
                7: float64, 8: uint16}

Data contract
-------------

- Returns ``{"input_ids": ..., "labels": ...}`` where ``labels == input_ids``.
  The model (:class:`GPT.forward`) performs the causal shift internally.
- Blocks are fixed-size ``block_size`` tokens.  Blocks never cross ``.bin``
  file boundaries — the tail of each file is discarded if shorter than
  ``block_size``.
- Deterministic, seed-controlled shuffling.
- Distributed-sampler aware: each rank sees a disjoint subset.
"""

from __future__ import annotations

import glob
import os
import random
import struct
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# LITPKDS header
# ---------------------------------------------------------------------------

HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24

DTYPE_MAP: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.float32),
    7: np.dtype(np.float64),
    8: np.dtype(np.uint16),
}


def _read_header(path: str) -> tuple[np.dtype, int]:
    """Read LITPKDS header from a ``.bin`` file.

    Returns ``(numpy_dtype, packed_chunk_size)`` where *packed_chunk_size*
    is the token count of each packed chunk (NOT the total file tokens).
    """
    with open(path, "rb") as f:
        magic = f.read(len(HDR_MAGIC))
        if magic != HDR_MAGIC:
            raise ValueError(
                f"Bad LITPKDS magic in {path}: {magic!r} (expected {HDR_MAGIC!r})"
            )
        (version,) = struct.unpack("<Q", f.read(8))
        if version != 1:
            raise ValueError(f"Unsupported LITPKDS version {version} in {path}")
        (dtype_code,) = struct.unpack("<B", f.read(1))
        if dtype_code not in DTYPE_MAP:
            raise ValueError(
                f"Unknown LITPKDS dtype code {dtype_code} in {path}"
            )
        (packed_chunk_size,) = struct.unpack("<Q", f.read(8))
    return DTYPE_MAP[dtype_code], packed_chunk_size


def _file_num_tokens(path: str, dtype: np.dtype) -> int:
    """Total number of tokens in the payload of a LITPKDS ``.bin`` file."""
    payload_bytes = os.path.getsize(path) - HDR_SIZE
    if payload_bytes % dtype.itemsize != 0:
        raise ValueError(
            f"Payload of {path} ({payload_bytes} bytes) is not divisible "
            f"by dtype.itemsize ({dtype.itemsize}) — file may be truncated"
        )
    return payload_bytes // dtype.itemsize


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProLongPackedDataset(Dataset):
    """Map-style dataset over pre-packed ProLong-64k ``.bin`` files.

    Blocks are pre-scanned and shuffled at init time, so ``__getitem__``
    is O(1) and compatible with ``DataLoader`` + ``DistributedSampler``.

    Parameters
    ----------
    data_dir:
        Directory containing ``*.bin`` files.
    block_size:
        Sequence length of each sample (65536 for 64K training).
    seed:
        Shuffle seed.  Set to the same value across workers with different
        ``rank`` / ``world_size`` for deterministic distributed sampling.
    rank:
        Rank of the current process (0-based).  Only used when
        ``world_size > 1``.
    world_size:
        Total number of distributed processes.
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        seed: int = 42,
        preload: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.seed = seed

        self.filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not self.filenames:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")

        # Pre-scan headers: build global block index and preload tokens
        self._blocks: list[tuple[int, int]] = []  # (start_token, count)
        total_tokens = 0
        all_arrs: list[np.ndarray] = []
        file_token_offsets: dict[str, tuple[int, int, np.dtype]] = {}

        if preload:
            for path in self.filenames:
                dtype, _packed_chunk_size = _read_header(path)
                num_tokens = _file_num_tokens(path, dtype)
                arr = np.memmap(path, dtype=dtype, mode="r", offset=HDR_SIZE, shape=(num_tokens,))
                all_arrs.append(arr)
                file_token_offsets[path] = (total_tokens, num_tokens, dtype)
                total_tokens += num_tokens
            self._token_buffer = np.concatenate(all_arrs).astype(np.int64)
        else:
            for path in self.filenames:
                dtype, _packed_chunk_size = _read_header(path)
                num_tokens = _file_num_tokens(path, dtype)
                file_token_offsets[path] = (total_tokens, num_tokens, dtype)
                total_tokens += num_tokens
            self._token_buffer = None

        for path in self.filenames:
            offset, num_tokens, dtype = file_token_offsets[path]
            n_blocks = num_tokens // block_size
            for b in range(n_blocks):
                self._blocks.append((offset + b * block_size, block_size))

        # Shuffle block indices with fixed seed
        rng = random.Random(seed)
        rng.shuffle(self._blocks)

        self._total_tokens = total_tokens
        self._total_blocks = len(self._blocks)

    def __len__(self) -> int:
        return self._total_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start, length = self._blocks[idx]
        tokens = torch.from_numpy(
            self._token_buffer[start : start + length].copy()
        ).long()
        return {"input_ids": tokens, "labels": tokens.clone()}

    @property
    def total_tokens(self) -> int:
        """Total raw tokens across all ``.bin`` files (excluding tails)."""
        return self._total_tokens

    @property
    def total_blocks(self) -> int:
        """Total ``block_size`` blocks across all files (before rank sharding)."""
        return self._total_blocks

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"ProLongPackedDataset: {len(self.filenames)} files, "
            f"{self._total_tokens / 1e9:.2f}B raw tokens, "
            f"{self._total_blocks} blocks of {self.block_size}",
            f"  seed={self.seed}",
        ]
        if self.filenames:
            lines.append(f"  first file: {self.filenames[0]}")
            lines.append(f"  last file : {self.filenames[-1]}")
        return "\n".join(lines)
