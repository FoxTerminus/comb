"""Timing and memory helpers for benchmark inference."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

import torch


@contextmanager
def cuda_timer(device: torch.device) -> Iterator[dict[str, float]]:
    stats: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield stats
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stats["elapsed_s"] = time.perf_counter() - start


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
