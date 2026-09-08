#!/usr/bin/env python3
"""Prove a real CombLlama checkpoint reduces exactly to its frozen Llama."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from training.artifact_io import write_text_atomic


REPO = Path("/data3/junhaohu/comb")
SOURCE_PATHS = (
    "benchmarks/audit_no_context_equivalence.py",
    "comb/integration/hf/CombLlama.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    # The upstream Llama API returns BF16 logits while the released Comb
    # wrapper explicitly casts logits to FP32.  Hash one canonical numerical
    # representation so API dtype does not masquerade as a value mismatch.
    values = tensor.detach().float().contiguous().cpu()
    # torch.equal correctly treats +0 and -0 as the same numeric value.  Use
    # the same canonical representation so equal logits have equal evidence
    # hashes as well.
    values = torch.where(values == 0, torch.zeros_like(values), values)
    return hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()


def comparison(left: torch.Tensor, right: torch.Tensor) -> dict:
    same_shape = tuple(left.shape) == tuple(right.shape)
    same_dtype = left.dtype == right.dtype
    if not same_shape:
        return {
            "same_shape": False,
            "same_dtype": same_dtype,
            "exact_equal": False,
            "maximum_absolute_delta": None,
            "left_sha256": tensor_sha256(left),
            "right_sha256": tensor_sha256(right),
        }
    delta = (left.float() - right.float()).abs()
    return {
        "same_shape": True,
        "same_dtype": same_dtype,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "exact_equal": torch.equal(left, right),
        "maximum_absolute_delta": float(delta.max()),
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
    }


def gpu_identity() -> dict:
    physical_gpu = os.environ.get("COMB_BENCHMARK_PHYSICAL_GPU")
    if physical_gpu is None:
        return {
            "physical_gpu": None,
            "query_succeeded": False,
            "name_and_driver": None,
        }
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            physical_gpu,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "physical_gpu": physical_gpu,
        "query_succeeded": result.returncode == 0,
        "name_and_driver": result.stdout.strip() if result.returncode == 0 else None,
    }


def load_base(path: Path, device: torch.device):
    return LlamaForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().to(device)


def load_comb(path: Path, device: torch.device):
    return CombLlamaForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comb", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    for path in (args.comb, args.base, args.tokenizer):
        if not path.is_dir():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    prompt = "Explain in one sentence why cached documents can be reused."
    context = "A position-independent representation does not encode its placement."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    chunk_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    cross_attention_mask = torch.ones_like(chunk_ids)

    with torch.inference_mode():
        base = load_base(args.base.resolve(), device)
        base_logits = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        ).logits.cpu()
        del base
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        comb = load_comb(args.comb.resolve(), device)
        cross_calls = 0

        def count_cross_call(_module, _inputs, _output):
            nonlocal cross_calls
            cross_calls += 1

        hooks = [
            layer.register_forward_hook(count_cross_call)
            for layer in comb.language_model.model.cross_layers
        ]
        try:
            no_context_before = comb(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            ).logits.cpu()
            calls_after_first_no_context = cross_calls
            comb(
                input_ids=input_ids,
                chunk_ids=chunk_ids,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            calls_after_context = cross_calls
            no_context_after = comb(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            ).logits.cpu()
            calls_after_second_no_context = cross_calls
        finally:
            for hook in hooks:
                hook.remove()

    base_vs_before = comparison(base_logits, no_context_before)
    before_vs_after = comparison(no_context_before, no_context_after)
    expected_cross_layers = len(comb.language_model.model.cross_layers)
    checks = {
        "base_vs_comb_no_context_exact": base_vs_before["exact_equal"],
        "no_stale_context_state": before_vs_after["exact_equal"],
        "first_no_context_skipped_cross_attention": calls_after_first_no_context == 0,
        "context_called_every_cross_layer_once": calls_after_context
        == expected_cross_layers,
        "second_no_context_skipped_cross_attention": calls_after_second_no_context
        == expected_cross_layers,
    }
    report = {
        "scope": "real checkpoint non-intrusive no-context equivalence",
        "comb_checkpoint": str(args.comb.resolve()),
        "base_checkpoint": str(args.base.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "prompt_token_ids": input_ids.cpu().tolist(),
        "context_token_ids": chunk_ids.cpu().tolist(),
        "weight_dtype": "torch.bfloat16",
        "attention_implementation": "eager",
        "base_vs_comb_no_context": base_vs_before,
        "comb_no_context_before_vs_after_context": before_vs_after,
        "cross_attention_calls": {
            "expected_layers": expected_cross_layers,
            "after_first_no_context": calls_after_first_no_context,
            "after_context": calls_after_context,
            "after_second_no_context": calls_after_second_no_context,
        },
        "hardware": gpu_identity(),
        "source_sha256": {
            relative: sha256_file(REPO / relative) for relative in SOURCE_PATHS
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_text_atomic(args.output, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("real checkpoint no-context equivalence failed")


if __name__ == "__main__":
    main()
