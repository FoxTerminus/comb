"""Adapter factory."""

from __future__ import annotations

from typing import Any

from benchmarks.scripts.adapters.base import BenchmarkAdapter
from benchmarks.scripts.adapters.diagnostics import diagnose_adapter_config, raise_for_diagnostics
from benchmarks.scripts.adapters.hf import (
    CombLlamaAdapter,
    LlamaAdapter,
    LoadErrorAdapter,
    SambaYAdapter,
    YOCOAdapter,
)
from benchmarks.scripts.adapters.mock import MockAdapter


def build_adapter(config: dict[str, Any]) -> BenchmarkAdapter:
    adapter = str(config.get("adapter", config.get("name", "mock"))).lower()
    if adapter == "mock":
        return MockAdapter(config)
    adapter_cls = {
        "llama": LlamaAdapter,
        "combllama": CombLlamaAdapter,
        "comb_llama": CombLlamaAdapter,
        "yoco": YOCOAdapter,
        "sambay": SambaYAdapter,
    }.get(adapter)
    if adapter_cls is None:
        raise ValueError(f"Unsupported adapter: {adapter}")
    try:
        raise_for_diagnostics(diagnose_adapter_config(config))
        return adapter_cls(config)
    except BaseException as exc:
        return LoadErrorAdapter(config, exc)
