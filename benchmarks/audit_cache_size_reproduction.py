#!/usr/bin/env python3
"""Record the exact configuration values underlying paper Figure 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from comb.supported_models import COMB_MODEL_MAPPING
from training.artifact_io import write_text_atomic


MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
)


def cache_record(model_name: str, base_config, comb_config, num_tokens: int) -> dict:
    bytes_per_element = base_config.dtype.itemsize
    single_layer_mb = (
        2
        * num_tokens
        * base_config.num_key_value_heads
        * base_config.head_dim
        * bytes_per_element
        / 2**20
    )
    cross_layers = list(comb_config.cross_attention_layers)
    baseline_mb = single_layer_mb * base_config.num_hidden_layers
    comb_mb = single_layer_mb * len(cross_layers)
    return {
        "model": model_name,
        "comb_model": COMB_MODEL_MAPPING[model_name],
        "num_tokens": num_tokens,
        "dtype": str(base_config.dtype),
        "bytes_per_element": bytes_per_element,
        "num_key_value_heads": base_config.num_key_value_heads,
        "head_dim": base_config.head_dim,
        "decoder_layers": base_config.num_hidden_layers,
        "cross_attention_layers": cross_layers,
        "baseline_cache_mb": baseline_mb,
        "comb_cache_mb": comb_mb,
        "retained_fraction": comb_mb / baseline_mb,
        "compression_factor": baseline_mb / comb_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/cache_size/cache_size_values.json"),
    )
    args = parser.parse_args()
    records = [
        cache_record(
            name,
            AutoConfig.from_pretrained(name, local_files_only=True),
            AutoConfig.from_pretrained(COMB_MODEL_MAPPING[name], local_files_only=True),
            args.num_tokens,
        )
        for name in MODELS
    ]
    result = {
        "scope": "paper Figure 8 KV-cache size calculation",
        "formula": (
            "2 * tokens * num_key_value_heads * head_dim * "
            "bytes_per_element * retained_layers"
        ),
        "records": records,
        "passed": all(
            record["comb_cache_mb"] < record["baseline_cache_mb"]
            for record in records
        ),
    }
    write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("COMB cache is not smaller than the baseline cache")


if __name__ == "__main__":
    main()
