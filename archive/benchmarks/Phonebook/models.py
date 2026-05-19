"""Generation adapters for Phonebook models."""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Any

from config import MODEL_SPECS, ModelSpec
from data import PhonebookExample, apply_chat_template


def clean_prediction(text: str) -> str:
    text = text.strip()
    lower = text.lower()
    if "</think>" in lower:
        text = text[lower.rfind("</think>") + len("</think>") :].strip()
    elif lower.startswith("<think>"):
        text = text.split("\n", 1)[-1].strip() if "\n" in text else ""
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _load_tokenizer(tokenizer_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _torch_dtype(dtype: str):
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype]


class BaseAdapter:
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 16,
        repetition_penalty: float = 1.0,
    ) -> None:
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.tokenizer = _load_tokenizer(spec.tokenizer_path)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, example: PhonebookExample) -> dict[str, Any]:
        raise NotImplementedError

    def apply_repetition_penalty(self, logits: Any, generated_ids: list[int]) -> Any:
        if self.repetition_penalty == 1.0 or not generated_ids:
            return logits
        for token_id in set(generated_ids):
            score = logits[:, token_id]
            logits[:, token_id] = self.torch.where(
                score < 0,
                score * self.repetition_penalty,
                score / self.repetition_penalty,
            )
        return logits


class HFCausalLMAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.spec.path),
            torch_dtype=_torch_dtype(self.dtype),
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, example: PhonebookExample) -> dict[str, Any]:
        torch = self.torch
        prompt = apply_chat_template(self.tokenizer, example.full_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        new_ids = output_ids[0, input_ids.shape[1] :]
        return {
            "prediction": clean_prediction(self.tokenizer.decode(new_ids, skip_special_tokens=True)),
            "input_tokens": int(input_ids.shape[1]),
            "chunk_tokens": 0,
            "output_tokens": int(new_ids.numel()),
            "latency_sec": latency,
        }


class CombQwenAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import torch
        from safetensors.torch import load_file

        repo_root = Path(__file__).resolve().parents[2]
        package_name = "_phonebook_comb_models"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(repo_root / "models")]
            sys.modules[package_name] = package

        from _phonebook_comb_models.comb_qwen import CombForConditionalGeneration
        from _phonebook_comb_models.config import CombConfig

        self.torch = torch
        config = CombConfig.from_pretrained(str(self.spec.path), trust_remote_code=True)
        self.model = CombForConditionalGeneration(config, from_scratch=False)
        state = load_file(str(self.spec.path / "model.safetensors"), device="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device, dtype=_torch_dtype(self.dtype))
        self.model.eval()

    def _encode_chunk(self, chunk_ids: Any):
        torch = self.torch
        k_len = chunk_ids.shape[1]
        position_ids_k = torch.arange(k_len, device=self.device, dtype=torch.long).unsqueeze(0)
        cu_chunk = torch.tensor([0, k_len], dtype=torch.int32, device=self.device)
        with torch.inference_mode():
            return self.model.chunk_model(
                chunk_ids,
                position_ids_k=position_ids_k,
                cu_seqlens_chunk=cu_chunk,
                max_seqlen_chunk=k_len,
            )

    def generate(self, example: PhonebookExample) -> dict[str, Any]:
        torch = self.torch
        chunk_ids = self.tokenizer(
            example.conversation_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        cross_attention_states = self._encode_chunk(chunk_ids)
        prompt = apply_chat_template(self.tokenizer, example.question_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        k_len = chunk_ids.shape[1]
        cu_k = torch.tensor([0, k_len], dtype=torch.int32, device=self.device)
        started = time.perf_counter()
        generated: list[int] = []
        cur_ids = input_ids
        with torch.inference_mode():
            for _ in range(self.max_new_tokens):
                q_len = cur_ids.shape[1]
                position_ids = torch.arange(q_len, device=self.device, dtype=torch.long).unsqueeze(0)
                cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=self.device)
                outputs = self.model.language_model.model(
                    input_ids=cur_ids,
                    cross_attention_states=cross_attention_states,
                    position_ids=position_ids,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=q_len,
                    max_seqlen_k=k_len,
                )
                logits = self.model.language_model.lm_head(outputs.last_hidden_state[:, -1, :])
                logits = self.apply_repetition_penalty(logits, generated)
                next_id = int(torch.argmax(logits, dim=-1).item())
                if next_id == self.tokenizer.eos_token_id:
                    break
                generated.append(next_id)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_id]], device=self.device, dtype=cur_ids.dtype)],
                    dim=1,
                )
        latency = time.perf_counter() - started
        return {
            "prediction": clean_prediction(self.tokenizer.decode(generated, skip_special_tokens=True)),
            "input_tokens": int(input_ids.shape[1]),
            "chunk_tokens": int(k_len),
            "output_tokens": len(generated),
            "latency_sec": latency,
        }


class SambaAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import torch
        from safetensors.torch import load_file

        repo_root = Path(__file__).resolve().parents[2]
        archscale_root = repo_root / "baselines" / "ArchScale"
        for path in (repo_root, archscale_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from baselines.ArchScale.models.config import Config
        from baselines.ArchScale.models.model import GPT

        self.torch = torch
        cfg = json.loads((self.spec.path / "config.json").read_text(encoding="utf-8"))
        cfg.pop("model_type", None)
        self.model = GPT(Config(**cfg))
        state = load_file(str(self.spec.path / "model.safetensors"), device="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device, dtype=_torch_dtype(self.dtype))
        self.model.eval()

    def generate(self, example: PhonebookExample) -> dict[str, Any]:
        torch = self.torch
        prompt = apply_chat_template(self.tokenizer, example.full_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        started = time.perf_counter()
        generated: list[int] = []
        cur_ids = input_ids
        with torch.inference_mode():
            for _ in range(self.max_new_tokens):
                outputs = self.model(cur_ids)
                logits = outputs["logits"][:, -1, :]
                logits = self.apply_repetition_penalty(logits, generated)
                next_id = int(torch.argmax(logits, dim=-1).item())
                if next_id == self.tokenizer.eos_token_id:
                    break
                generated.append(next_id)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[next_id]], device=self.device, dtype=cur_ids.dtype)],
                    dim=1,
                )
        latency = time.perf_counter() - started
        return {
            "prediction": clean_prediction(self.tokenizer.decode(generated, skip_special_tokens=True)),
            "input_tokens": int(input_ids.shape[1]),
            "chunk_tokens": 0,
            "output_tokens": len(generated),
            "latency_sec": latency,
        }


def load_adapter(
    model_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 16,
    repetition_penalty: float = 1.0,
) -> BaseAdapter:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {model_name!r}. Known: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[model_name]
    kwargs = dict(
        spec=spec,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    if spec.kind == "hf_causal_lm":
        return HFCausalLMAdapter(**kwargs)
    if spec.kind == "comb_qwen":
        return CombQwenAdapter(**kwargs)
    if spec.kind == "samba":
        return SambaAdapter(**kwargs)
    raise ValueError(f"Unsupported model kind: {spec.kind}")
