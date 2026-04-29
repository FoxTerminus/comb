"""Base model adapter interfaces."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from benchmarks.scripts.schema import BenchmarkExample, GenerationRecord


def render_prompt(example: BenchmarkExample) -> str:
    choices = ""
    if example.choices:
        choices = "\nChoices: " + ", ".join(example.choices)
    return (
        "Use the context to answer the question. Keep the answer concise.\n\n"
        f"Context:\n{example.context}\n\n"
        f"Question:\n{example.question}{choices}\n\n"
        "Answer:"
    )


def basic_token_count(text: str) -> int:
    return len(text.split())


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_s = time.perf_counter() - self.start


def reset_peak_memory(device: Any) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def peak_memory_gb(device: Any) -> float | None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device) / (1024**3)
    except Exception:
        return None
    return None


class BenchmarkAdapter(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_name = str(config.get("name", "unknown"))
        self.checkpoint = str(config.get("model_path", ""))

    @abstractmethod
    def generate(self, example: BenchmarkExample, generation_config: dict[str, Any], run_id: str) -> GenerationRecord:
        raise NotImplementedError

