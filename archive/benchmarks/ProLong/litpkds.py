"""Helpers for LITPKDS token files used by ProLong."""

from __future__ import annotations

import glob
import os
import random
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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

DTYPE_REV = {np.dtype("uint16"): 8, np.dtype("int32"): 4, np.dtype("int64"): 5}


def read_header(path: str | Path) -> tuple[np.dtype, int]:
    with Path(path).open("rb") as f:
        magic = f.read(len(HDR_MAGIC))
        if magic != HDR_MAGIC:
            raise ValueError(f"Bad LITPKDS magic in {path}: {magic!r}")
        version = struct.unpack("<Q", f.read(8))[0]
        if version != 1:
            raise ValueError(f"Unsupported LITPKDS version {version} in {path}")
        dtype_code = struct.unpack("<B", f.read(1))[0]
        if dtype_code not in DTYPE_MAP:
            raise ValueError(f"Unknown LITPKDS dtype code {dtype_code} in {path}")
        packed_chunk_size = struct.unpack("<Q", f.read(8))[0]
    return DTYPE_MAP[dtype_code], packed_chunk_size


def file_num_tokens(path: str | Path, dtype: np.dtype | None = None) -> int:
    dtype = dtype or read_header(path)[0]
    payload_bytes = os.path.getsize(path) - HDR_SIZE
    if payload_bytes % dtype.itemsize != 0:
        raise ValueError(f"Truncated payload in {path}")
    return payload_bytes // dtype.itemsize


def write_litpkds(path: str | Path, tokens: list[int] | np.ndarray, dtype=np.int32) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    if dtype not in DTYPE_REV:
        raise ValueError(f"Unsupported output dtype {dtype}")
    arr = np.asarray(tokens, dtype=dtype)
    with path.open("wb") as f:
        f.write(HDR_MAGIC)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", DTYPE_REV[dtype]))
        f.write(struct.pack("<Q", len(arr)))
        f.write(arr.tobytes())


@dataclass(frozen=True)
class BlockRef:
    path: str
    start: int
    length: int
    dtype: np.dtype


class LitpkdsBlockDataset(Dataset):
    """Map-style dataset over fixed-length windows from LITPKDS files."""

    def __init__(
        self,
        data_dir: str | Path,
        block_size: int,
        pattern: str = "validation*.bin",
        seed: int = 42,
        max_samples: int | None = None,
        file_start: int = 0,
        file_stride: int = 20,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        all_filenames = sorted(glob.glob(str(self.data_dir / pattern)))
        self.filenames = [
            path for path in all_filenames
            if file_id(path) is not None
            and (file_id(path) - file_start) % file_stride == 0
        ]
        if not self.filenames:
            raise FileNotFoundError(
                f"No files match {self.data_dir / pattern} "
                f"with file_start={file_start}, file_stride={file_stride}"
            )

        blocks: list[BlockRef] = []
        total_tokens = 0
        for path in self.filenames:
            dtype, _ = read_header(path)
            num_tokens = file_num_tokens(path, dtype)
            for start in range(0, num_tokens - block_size + 1, block_size):
                blocks.append(BlockRef(path, start, block_size, dtype))
            total_tokens += num_tokens

        random.Random(seed).shuffle(blocks)
        if max_samples is not None:
            blocks = blocks[:max_samples]
        self.blocks = blocks
        self.total_tokens = total_tokens

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.blocks[idx]
        arr = np.memmap(ref.path, dtype=ref.dtype, mode="r", offset=HDR_SIZE)
        tokens = torch.from_numpy(
            arr[ref.start : ref.start + ref.length].astype(np.int64).copy()
        ).long()
        return {"input_ids": tokens, "labels": tokens.clone()}


def file_id(path: str | Path) -> int | None:
    """Return the numeric id at the end of a ProLong file stem."""
    match = re.search(r"_(\d+)$", Path(path).stem)
    return int(match.group(1)) if match else None
