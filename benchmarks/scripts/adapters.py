"""Model adapters used by benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from benchmarks.scripts.instrumentation import cuda_timer, peak_memory_gb, reset_peak_memory
from benchmarks.scripts.prompting import pack_combllama_prompt, render_qa_prompt
from benchmarks.scripts.schema import BenchmarkExample, GenerationRecord


@dataclass
class CachedCombContext:
    prompt_tokens: int
    chunk_tokens: int
    initial_decoder_tokens: int
    cross_attention_states: list[tuple[torch.Tensor, torch.Tensor]] | None
    cu_seqlens_k: torch.Tensor | None
    max_seqlen_k: int | None


class BenchmarkAdapter(ABC):
    @abstractmethod
    def generate(self, example: BenchmarkExample, max_new_tokens: int) -> GenerationRecord:
        raise NotImplementedError


class MockAdapter(BenchmarkAdapter):
    """Tiny deterministic adapter for validating benchmark plumbing."""

    def __init__(self, model_name: str = "mock-combllama") -> None:
        self.model_name = model_name

    def generate(self, example: BenchmarkExample, max_new_tokens: int) -> GenerationRecord:
        del max_new_tokens
        prediction = example.answer or "mock answer"
        return GenerationRecord(
            id=example.id,
            benchmark=example.benchmark,
            task=example.task,
            model=self.model_name,
            prompt_tokens=len((example.context + example.question).split()),
            chunk_tokens=0,
            decoder_tokens=0,
            generated_tokens=len(prediction.split()),
            compression_ratio=1.0,
            prediction=prediction,
            answer=example.answer,
            exact_match=True if example.answer is not None else None,
            peak_memory_gb=None,
            prefill_latency_s=0.0,
            decode_latency_s=0.0,
            tokens_per_second=None,
        )


class CombLlamaAdapter(BenchmarkAdapter):
    """Greedy CombLlama inference adapter for benchmark evaluation.

    The adapter packs old prompt tokens into `chunk_ids` and keeps a recent
    decoder window in `input_ids`. By default it caches the chunk encoder output
    once per prompt, then reuses those cross-attention states for every generated
    token. This avoids repeatedly encoding 10K-100K historical tokens while
    still not relying on the unverified self-attention KV cache path.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoTokenizer

        from models.CombLlama import CombLlamaForConditionalGeneration

        self.config = config
        self.model_name = str(config.get("name", "combllama"))
        self.model_path = str(config["model_path"])
        tokenizer_path = str(config.get("tokenizer_path", self.model_path))
        self.chunk_size = int(config.get("chunk_size", 1024))
        self.recent_window_tokens = int(config.get("recent_window_tokens", 1024))
        self.compression_ratio = float(config.get("compression_ratio", 1.0))
        self.retention_policy = str(config.get("retention_policy", "all_encoder_chunks"))
        if self.retention_policy != "all_encoder_chunks":
            raise ValueError("Only all_encoder_chunks is supported for CombLlama benchmarks")
        self.use_chat_template = bool(config.get("use_chat_template", True))
        self.cache_chunk_states = bool(config.get("cache_chunk_states", True))

        requested_device = str(config.get("device", "cuda"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(requested_device)

        dtype_name = str(config.get("dtype", "bfloat16"))
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = CombLlamaForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            self.eos_token_ids: set[int] = set()
        elif isinstance(eos_token_id, list):
            self.eos_token_ids = {int(token_id) for token_id in eos_token_id}
        else:
            self.eos_token_ids = {int(eos_token_id)}

    def _encode_prompt(self, prompt: str) -> list[int]:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def _answer_token_start(self, prompt: str, token_ids: list[int], answer: str | None) -> int | None:
        del prompt, token_ids, answer
        return None

    def _move_tensors(self, tensors: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tensors.items()
        }

    def _forward_last_logits(
        self,
        token_ids: list[int],
        answer_token_start: int | None = None,
    ) -> tuple[torch.Tensor, int, int, int]:
        packed = pack_combllama_prompt(
            token_ids,
            chunk_size=self.chunk_size,
            recent_window_tokens=self.recent_window_tokens,
            compression_ratio=self.compression_ratio,
            retention_policy=self.retention_policy,
            answer_token_start=answer_token_start,
        )
        inputs = self._move_tensors(packed.tensors)
        outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
        return (
            outputs.logits[:, -1, :],
            packed.prompt_tokens,
            packed.chunk_tokens,
            packed.decoder_tokens,
        )

    def _prepare_cached_context(
        self,
        token_ids: list[int],
        answer_token_start: int | None = None,
    ) -> tuple[CachedCombContext, dict[str, Any]]:
        packed = pack_combllama_prompt(
            token_ids,
            chunk_size=self.chunk_size,
            recent_window_tokens=self.recent_window_tokens,
            compression_ratio=self.compression_ratio,
            retention_policy=self.retention_policy,
            answer_token_start=answer_token_start,
        )
        inputs = self._move_tensors(packed.tensors)
        cross_attention_states = None
        cu_seqlens_k = None
        max_seqlen_k = None

        if "chunk_ids" in inputs:
            cross_attention_states = self.model.chunk_model(
                inputs["chunk_ids"],
                inputs["cu_seqlens_chunk"],
                inputs["max_seqlen_chunk"],
                inputs["position_ids_k"],
            )
            cu_seqlens_k = inputs["cu_seqlens_k"]
            max_seqlen_k = inputs["max_seqlen_k"]

        context = CachedCombContext(
            prompt_tokens=packed.prompt_tokens,
            chunk_tokens=packed.chunk_tokens,
            initial_decoder_tokens=packed.decoder_tokens,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )
        decoder_inputs = self._decoder_inputs(token_ids)
        return context, decoder_inputs

    def _decoder_inputs(self, token_ids: list[int]) -> dict[str, Any]:
        decoder_ids = token_ids[-max(1, self.recent_window_tokens):]
        tensors = {
            "input_ids": torch.tensor([decoder_ids], dtype=torch.long),
            "position_ids": torch.arange(len(decoder_ids), dtype=torch.long).unsqueeze(0),
            "cu_seqlens_q": torch.tensor([0, len(decoder_ids)], dtype=torch.int32),
            "max_seqlen_q": len(decoder_ids),
        }
        return self._move_tensors(tensors)

    def _forward_cached_last_logits(
        self,
        token_ids: list[int],
        context: CachedCombContext,
    ) -> tuple[torch.Tensor, int, int, int]:
        inputs = self._decoder_inputs(token_ids)
        if context.cross_attention_states is not None:
            inputs.update(
                {
                    "cross_attention_states": context.cross_attention_states,
                    "cu_seqlens_k": context.cu_seqlens_k,
                    "max_seqlen_k": context.max_seqlen_k,
                }
            )
        outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
        return (
            outputs.logits[:, -1, :],
            context.prompt_tokens,
            context.chunk_tokens,
            inputs["max_seqlen_q"],
        )

    def generate(self, example: BenchmarkExample, max_new_tokens: int) -> GenerationRecord:
        prompt = render_qa_prompt(example)
        token_ids = self._encode_prompt(prompt)
        if not token_ids:
            token_ids = [next(iter(self.eos_token_ids), 0)]
        answer_token_start = self._answer_token_start(prompt, token_ids, example.answer)

        generated: list[int] = []
        prompt_tokens = len(token_ids)
        chunk_tokens = 0
        decoder_tokens = 0
        reset_peak_memory(self.device)

        with torch.inference_mode():
            with cuda_timer(self.device) as prefill_timer:
                if self.cache_chunk_states:
                    cached_context, _ = self._prepare_cached_context(token_ids, answer_token_start)
                    logits, prompt_tokens, chunk_tokens, decoder_tokens = self._forward_cached_last_logits(
                        token_ids,
                        cached_context,
                    )
                else:
                    cached_context = None
                    logits, prompt_tokens, chunk_tokens, decoder_tokens = self._forward_last_logits(
                        token_ids,
                        answer_token_start,
                    )
                next_token = int(torch.argmax(logits, dim=-1).item())

            with cuda_timer(self.device) as decode_timer:
                for step in range(max_new_tokens):
                    if step > 0:
                        if cached_context is not None:
                            logits, prompt_tokens, chunk_tokens, decoder_tokens = self._forward_cached_last_logits(
                                token_ids + generated,
                                cached_context,
                            )
                        else:
                            logits, prompt_tokens, chunk_tokens, decoder_tokens = self._forward_last_logits(
                                token_ids + generated,
                                answer_token_start,
                            )
                        next_token = int(torch.argmax(logits, dim=-1).item())
                    if next_token in self.eos_token_ids:
                        break
                    generated.append(next_token)

        prediction = self.tokenizer.decode(generated, skip_special_tokens=True)
        decode_latency = decode_timer["elapsed_s"]
        tokens_per_second = None
        if generated and decode_latency > 0:
            tokens_per_second = len(generated) / decode_latency

        normalized_prediction = prediction.strip().lower()
        normalized_answer = None if example.answer is None else example.answer.strip().lower()
        exact_match = None
        if normalized_answer is not None:
            exact_match = normalized_prediction == normalized_answer

        return GenerationRecord(
            id=example.id,
            benchmark=example.benchmark,
            task=example.task,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            chunk_tokens=chunk_tokens,
            decoder_tokens=decoder_tokens,
            generated_tokens=len(generated),
            compression_ratio=self.compression_ratio,
            prediction=prediction,
            answer=example.answer,
            exact_match=exact_match,
            peak_memory_gb=peak_memory_gb(self.device),
            prefill_latency_s=prefill_timer["elapsed_s"],
            decode_latency_s=decode_latency,
            tokens_per_second=tokens_per_second,
        )
