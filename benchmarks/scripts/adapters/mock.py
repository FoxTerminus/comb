"""Deterministic adapter for validating the benchmark framework."""

from __future__ import annotations

from typing import Any

from benchmarks.scripts.adapters.base import BenchmarkAdapter, basic_token_count, render_prompt
from benchmarks.scripts.schema import BenchmarkExample, GenerationRecord


class MockAdapter(BenchmarkAdapter):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config or {"name": "mock", "model_path": "mock"})

    def generate(self, example: BenchmarkExample, generation_config: dict[str, Any], run_id: str) -> GenerationRecord:
        del generation_config
        prompt = render_prompt(example)
        prediction = example.answer or "mock answer"
        return GenerationRecord(
            run_id=run_id,
            model=self.model_name,
            checkpoint=self.checkpoint,
            benchmark=example.benchmark,
            task=example.task,
            id=example.id,
            prediction=prediction,
            answer=example.answer,
            metrics={},
            example_metadata=example.metadata,
            prompt_tokens=basic_token_count(prompt),
            context_tokens=basic_token_count(example.context),
            generated_tokens=basic_token_count(prediction),
            kv_cache_policy=str(self.config.get("kv_cache_policy", "mock_no_kv_cache")),
            chunk_size=self.config.get("chunk_size"),
            recent_window_tokens=self.config.get("recent_window_tokens"),
            peak_memory_gb=None,
            prefill_latency_s=0.0,
            decode_latency_s=0.0,
            tokens_per_second=None,
            error=None,
        )
