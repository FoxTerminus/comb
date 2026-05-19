"""Model and benchmark defaults for LoCoMo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


ROOT = Path("/data3/junhaohu")
BENCHMARK_DIR = ROOT / "comb" / "benchmarks" / "LoCoMo"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"
DEFAULT_DATA_FILE = (
    ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--Percena--locomo-mc10"
    / "snapshots"
    / "7d59a0463d83f97b042684310c0b3d17553004cd"
    / "raw"
    / "locomo10.json"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path
    tokenizer_path: Union[str, Path]
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
    "comb-llama-8b-instruct": ModelSpec(
        name="comb-llama-8b-instruct",
        kind="comb_llama",
        path=ROOT / "model" / "CombLlama-8B-Instruct",
        tokenizer_path="meta-llama/Llama-3.1-8B-Instruct",
        trained_on_prolong_64k=False,
        max_context_tokens=131072,
    ),
}


DEFAULT_MODELS = ("comb-qwen", "sambayoco", "sambay", "qwen3-0.6b")
ANS_TOKENS_PER_QUESTION = 50
