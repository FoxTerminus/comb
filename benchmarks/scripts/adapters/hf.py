"""Adapters backed by local HuggingFace-style models."""

from __future__ import annotations

from typing import Any

from benchmarks.scripts.adapters.base import (
    BenchmarkAdapter,
    Timer,
    basic_token_count,
    peak_memory_gb,
    render_prompt,
    reset_peak_memory,
)
from benchmarks.scripts.schema import BenchmarkExample, GenerationRecord


class LoadErrorAdapter(BenchmarkAdapter):
    """Adapter returned when model construction fails."""

    def __init__(self, config: dict[str, Any], error: BaseException) -> None:
        super().__init__(config)
        self.error = error

    def generate(self, example: BenchmarkExample, generation_config: dict[str, Any], run_id: str) -> GenerationRecord:
        del generation_config
        return GenerationRecord(
            run_id=run_id,
            model=self.model_name,
            checkpoint=self.checkpoint,
            benchmark=example.benchmark,
            task=example.task,
            id=example.id,
            prediction="",
            answer=example.answer,
            metrics={},
            example_metadata=example.metadata,
            prompt_tokens=basic_token_count(render_prompt(example)),
            context_tokens=basic_token_count(example.context),
            generated_tokens=0,
            kv_cache_policy=str(self.config.get("kv_cache_policy", "unknown")),
            chunk_size=self.config.get("chunk_size"),
            recent_window_tokens=self.config.get("recent_window_tokens"),
            peak_memory_gb=None,
            prefill_latency_s=None,
            decode_latency_s=None,
            tokens_per_second=None,
            error=f"ModelLoadError: {type(self.error).__name__}: {self.error}",
        )


class CausalLMAdapter(BenchmarkAdapter):
    model_cls: Any = None

    def __init__(self, config: dict[str, Any], model_cls: Any | None = None) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_cls = model_cls or AutoModelForCausalLM
        self.tokenizer_path = str(config.get("tokenizer_path", self.checkpoint))
        self.device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[str(config.get("dtype", "bfloat16"))]
        self.max_context_tokens = int(config.get("max_context_tokens", 131072))
        self.truncation_policy = str(config.get("sliding_or_truncation_policy", "left_truncate_when_required"))
        self.use_chat_template = bool(config.get("use_chat_template", True))
        self.use_cache = bool(config.get("use_cache", True))

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = self.model_cls.from_pretrained(self.checkpoint, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

    def _encode_prompt(self, prompt: str) -> list[int]:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            token_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(token_ids) <= self.max_context_tokens:
            return token_ids
        if self.truncation_policy != "left_truncate_when_required":
            raise ValueError(f"Prompt has {len(token_ids)} tokens and truncation is disabled")
        return token_ids[-self.max_context_tokens :]

    def generate(self, example: BenchmarkExample, generation_config: dict[str, Any], run_id: str) -> GenerationRecord:
        prompt = render_prompt(example)
        token_ids = self._encode_prompt(prompt)
        input_ids = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
        max_new_tokens = int(generation_config.get("max_new_tokens", 32))
        reset_peak_memory(self.device)
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(generation_config.get("do_sample", False)),
            "top_p": float(generation_config.get("top_p", 1.0)),
            "use_cache": self.use_cache,
        }
        temperature = float(generation_config.get("temperature", 0.0))
        if generate_kwargs["do_sample"] and temperature > 0:
            generate_kwargs["temperature"] = temperature
        with self.torch.inference_mode():
            with Timer() as timer:
                outputs = self.model.generate(**generate_kwargs)
        generated_ids = outputs[0, input_ids.shape[1] :].tolist()
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        tokens_per_second = None if timer.elapsed_s <= 0 else len(generated_ids) / timer.elapsed_s
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
            prompt_tokens=len(token_ids),
            context_tokens=basic_token_count(example.context),
            generated_tokens=len(generated_ids),
            kv_cache_policy=str(self.config.get("kv_cache_policy", "full_decoder_kv_cache")),
            chunk_size=self.config.get("chunk_size"),
            recent_window_tokens=self.config.get("recent_window_tokens"),
            peak_memory_gb=peak_memory_gb(self.device),
            prefill_latency_s=timer.elapsed_s,
            decode_latency_s=timer.elapsed_s,
            tokens_per_second=tokens_per_second,
            error=None,
        )


class LlamaAdapter(CausalLMAdapter):
    pass


class YOCOAdapter(CausalLMAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        from baselines.YOCO.models.YOCO import YOCOForCausalLM

        super().__init__(config, model_cls=YOCOForCausalLM)


class SambaYAdapter(CausalLMAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        from baselines.SambaY.models.SambaY import SambaYForCausalLM

        super().__init__(config, model_cls=SambaYForCausalLM)


class CombLlamaAdapter(BenchmarkAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoTokenizer

        from models.CombLlama import CombLlamaForConditionalGeneration

        self.torch = torch
        self.tokenizer_path = str(config.get("tokenizer_path", self.checkpoint))
        self.device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[str(config.get("dtype", "bfloat16"))]
        self.chunk_size = int(config.get("chunk_size", 1024))
        self.recent_window_tokens = int(config.get("recent_window_tokens", 1024))
        self.kv_cache_policy = str(config.get("kv_cache_policy", "chunk_encoder_cross_attention_kv"))
        self.use_chat_template = bool(config.get("use_chat_template", True))
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = CombLlamaForConditionalGeneration.from_pretrained(self.checkpoint, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        eos = self.tokenizer.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))

    def _encode_prompt(self, prompt: str) -> list[int]:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def _pack_inputs(self, token_ids: list[int]) -> dict[str, Any]:
        torch = self.torch
        recent = token_ids[-max(1, self.recent_window_tokens) :]
        old = token_ids[: max(0, len(token_ids) - len(recent))]
        inputs: dict[str, Any] = {
            "input_ids": torch.tensor([recent], dtype=torch.long, device=self.device),
            "position_ids": torch.arange(len(recent), dtype=torch.long, device=self.device).unsqueeze(0),
            "cu_seqlens_q": torch.tensor([0, len(recent)], dtype=torch.int32, device=self.device),
            "max_seqlen_q": len(recent),
            "logits_to_keep": 1,
            "use_cache": False,
        }
        if old:
            chunks = [old[index : index + self.chunk_size] for index in range(0, len(old), self.chunk_size)]
            flat = [token for chunk in chunks for token in chunk]
            cu = [0]
            for chunk in chunks:
                cu.append(cu[-1] + len(chunk))
            inputs.update(
                {
                    "chunk_ids": torch.tensor([flat], dtype=torch.long, device=self.device),
                    "position_ids_k": torch.arange(len(flat), dtype=torch.long, device=self.device).unsqueeze(0),
                    "cu_seqlens_k": torch.tensor([0, len(flat)], dtype=torch.int32, device=self.device),
                    "max_seqlen_k": len(flat),
                    "cu_seqlens_chunk": torch.tensor(cu, dtype=torch.int32, device=self.device),
                    "max_seqlen_chunk": max(len(chunk) for chunk in chunks),
                }
            )
        return inputs

    def generate(self, example: BenchmarkExample, generation_config: dict[str, Any], run_id: str) -> GenerationRecord:
        prompt = render_prompt(example)
        token_ids = self._encode_prompt(prompt)
        max_new_tokens = int(generation_config.get("max_new_tokens", 32))
        generated: list[int] = []
        reset_peak_memory(self.device)
        with self.torch.inference_mode():
            with Timer() as timer:
                for _ in range(max_new_tokens):
                    outputs = self.model(**self._pack_inputs(token_ids + generated))
                    next_token = int(self.torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                    if next_token in self.eos_token_ids:
                        break
                    generated.append(next_token)
        prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        tokens_per_second = None if timer.elapsed_s <= 0 else len(generated) / timer.elapsed_s
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
            prompt_tokens=len(token_ids),
            context_tokens=basic_token_count(example.context),
            generated_tokens=len(generated),
            kv_cache_policy=self.kv_cache_policy,
            chunk_size=self.chunk_size,
            recent_window_tokens=self.recent_window_tokens,
            peak_memory_gb=peak_memory_gb(self.device),
            prefill_latency_s=timer.elapsed_s,
            decode_latency_s=timer.elapsed_s,
            tokens_per_second=tokens_per_second,
            error=None,
        )
