"""Defaults for the Phonebook retrieval benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/data3/junhaohu")
BENCHMARK_DIR = ROOT / "comb" / "benchmarks" / "Phonebook"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path
    tokenizer_path: Path
    trained_on_prolong_64k: bool
    max_context_tokens: int = 65536


MODEL_SPECS = {
    "comb-qwen": ModelSpec(
        name="comb-qwen",
        kind="comb_qwen",
        path=ROOT / "model" / "Comb-Qwen3-1B",
        tokenizer_path=ROOT / "model" / "Comb-Qwen3-1B",
        trained_on_prolong_64k=True,
        max_context_tokens=65536,
    ),
    "sambayoco": ModelSpec(
        name="sambayoco",
        kind="samba",
        path=ROOT / "model" / "SambaYOCO-1B",
        tokenizer_path=ROOT / "model" / "Llama-2-7b-hf-tokenizer",
        trained_on_prolong_64k=True,
        max_context_tokens=65536,
    ),
    "sambay": ModelSpec(
        name="sambay",
        kind="samba",
        path=ROOT / "model" / "SambaY-1B",
        tokenizer_path=ROOT / "model" / "Llama-2-7b-hf-tokenizer",
        trained_on_prolong_64k=True,
        max_context_tokens=65536,
    ),
    "qwen3-0.6b": ModelSpec(
        name="qwen3-0.6b",
        kind="hf_causal_lm",
        path=ROOT / "model" / "Qwen3-0.6B",
        tokenizer_path=ROOT / "model" / "Qwen3-0.6B",
        trained_on_prolong_64k=False,
        max_context_tokens=40960,
    ),
}

DEFAULT_MODELS = ("comb-qwen", "sambayoco", "sambay", "qwen3-0.6b")
DEFAULT_NUM_SAMPLES = 200
DEFAULT_NUM_PAIRS = 1850
DEFAULT_MAX_NEW_TOKENS = 16

