"""Initialize a YOCO-Llama checkpoint from a HuggingFace Llama checkpoint.

This script implements the Stage 4 mapping for the repository's pure
YOCO-Llama baseline:

- self-decoder layers <- Llama layers 0..15
- cross-decoder MLP / norms <- Llama layers 16..31
- cross-decoder attention projections <- Llama layers 16..31 self-attention projections
- embeddings / final norm / lm_head <- copied from Llama
"""

import argparse
import json
import os
import tempfile
from typing import Dict, Tuple


def _ensure_writable_runtime_cache() -> str:
    """Route runtime caches to a writable directory.

    On shared machines, the default HuggingFace / torch extension cache
    locations may point to directories the current user cannot write to. This
    helper redirects the most common cache env vars to a writable temp path when
    they are unset or not writable.
    """
    preferred_roots = ["/custom_tmp", "/tmp", tempfile.gettempdir()]
    base_root = next((root for root in preferred_roots if os.path.isdir(root) and os.access(root, os.W_OK)), None)
    if base_root is None:
        base_root = tempfile.gettempdir()

    runtime_cache_root = os.path.join(base_root, "yoco_runtime_cache")
    os.makedirs(runtime_cache_root, exist_ok=True)

    cache_env_map = {
        "HF_HOME": os.path.join(runtime_cache_root, "hf_home"),
        "HF_HUB_CACHE": os.path.join(runtime_cache_root, "hf_hub"),
        "TRANSFORMERS_CACHE": os.path.join(runtime_cache_root, "transformers"),
        "XDG_CACHE_HOME": os.path.join(runtime_cache_root, "xdg"),
        "TORCH_HOME": os.path.join(runtime_cache_root, "torch"),
        "TORCH_EXTENSIONS_DIR": os.path.join(runtime_cache_root, "torch_extensions"),
        "TRITON_CACHE_DIR": os.path.join(runtime_cache_root, "triton"),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }

    for env_name, fallback_dir in cache_env_map.items():
        if env_name in {"HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER"}:
            os.environ.setdefault(env_name, fallback_dir)
            continue
        current_value = os.environ.get(env_name)
        if current_value and os.path.isdir(current_value) and os.access(current_value, os.W_OK):
            continue
        os.makedirs(fallback_dir, exist_ok=True)
        os.environ[env_name] = fallback_dir

    return runtime_cache_root


_RUNTIME_CACHE_ROOT = _ensure_writable_runtime_cache()

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from baselines.YOCO.models.YOCO import YOCOConfig, YOCOForCausalLM


def count_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _copy_param(dst: torch.nn.Parameter, src: torch.Tensor):
    if dst.shape != src.shape:
        raise ValueError(f"Shape mismatch: dst={tuple(dst.shape)} src={tuple(src.shape)}")
    with torch.no_grad():
        dst.copy_(src)


def _copy_or_repeat_kv_param(
    dst: torch.nn.Parameter,
    src: torch.Tensor,
    dst_heads: int,
    src_heads: int,
):
    if dst.shape == src.shape:
        _copy_param(dst, src)
        return
    if dst_heads % src_heads != 0:
        raise ValueError(f"Cannot repeat KV heads: dst_heads={dst_heads}, src_heads={src_heads}")
    head_dim = src.shape[0] // src_heads
    expanded = src.view(src_heads, head_dim, src.shape[1]).repeat_interleave(
        dst_heads // src_heads,
        dim=0,
    )
    expanded = expanded.reshape(dst.shape)
    _copy_param(dst, expanded)


def copy_embedding_and_heads(yoco: YOCOForCausalLM, llama) -> None:
    _copy_param(yoco.model.embed_tokens.weight, llama.model.embed_tokens.weight.data)
    _copy_param(yoco.model.norm.weight, llama.model.norm.weight.data)
    _copy_param(yoco.lm_head.weight, llama.lm_head.weight.data)


def copy_self_decoder(yoco: YOCOForCausalLM, llama) -> None:
    for idx, yoco_layer in enumerate(yoco.model.self_decoder.layers):
        llama_layer = llama.model.layers[idx]

        _copy_param(yoco_layer.input_layernorm.weight, llama_layer.input_layernorm.weight.data)
        _copy_param(yoco_layer.post_attention_layernorm.weight, llama_layer.post_attention_layernorm.weight.data)

        _copy_param(yoco_layer.self_attn.q_proj.weight, llama_layer.self_attn.q_proj.weight.data)
        _copy_or_repeat_kv_param(
            yoco_layer.self_attn.k_proj.weight,
            llama_layer.self_attn.k_proj.weight.data,
            dst_heads=yoco_layer.self_attn.num_key_value_heads,
            src_heads=llama.config.num_key_value_heads,
        )
        _copy_or_repeat_kv_param(
            yoco_layer.self_attn.v_proj.weight,
            llama_layer.self_attn.v_proj.weight.data,
            dst_heads=yoco_layer.self_attn.num_key_value_heads,
            src_heads=llama.config.num_key_value_heads,
        )
        _copy_param(yoco_layer.self_attn.o_proj.weight, llama_layer.self_attn.o_proj.weight.data)

        _copy_param(yoco_layer.mlp.gate_proj.weight, llama_layer.mlp.gate_proj.weight.data)
        _copy_param(yoco_layer.mlp.up_proj.weight, llama_layer.mlp.up_proj.weight.data)
        _copy_param(yoco_layer.mlp.down_proj.weight, llama_layer.mlp.down_proj.weight.data)


def copy_cross_decoder(yoco: YOCOForCausalLM, llama) -> None:
    offset = yoco.config.num_self_decoder_layers
    first_cross_source = llama.model.layers[offset]
    _copy_param(
        yoco.model.cross_decoder.kv_layer_norm.weight,
        first_cross_source.input_layernorm.weight.data,
    )
    _copy_param(
        yoco.model.cross_decoder.k_proj.weight,
        first_cross_source.self_attn.k_proj.weight.data,
    )
    _copy_param(
        yoco.model.cross_decoder.v_proj.weight,
        first_cross_source.self_attn.v_proj.weight.data,
    )

    for idx, yoco_layer in enumerate(yoco.model.cross_decoder.layers):
        llama_layer = llama.model.layers[offset + idx]

        _copy_param(yoco_layer.input_layernorm.weight, llama_layer.input_layernorm.weight.data)
        _copy_param(yoco_layer.post_attention_layernorm.weight, llama_layer.post_attention_layernorm.weight.data)

        # Official YOCO shares K/V across all cross-decoder layers. Per-layer
        # cross-attention keeps only Q and output projections.
        _copy_param(yoco_layer.cross_attn.q_proj.weight, llama_layer.self_attn.q_proj.weight.data)
        _copy_param(yoco_layer.cross_attn.o_proj.weight, llama_layer.self_attn.o_proj.weight.data)

        _copy_param(yoco_layer.mlp.gate_proj.weight, llama_layer.mlp.gate_proj.weight.data)
        _copy_param(yoco_layer.mlp.up_proj.weight, llama_layer.mlp.up_proj.weight.data)
        _copy_param(yoco_layer.mlp.down_proj.weight, llama_layer.mlp.down_proj.weight.data)


def build_yoco_from_llama(
    llama_path: str,
    torch_dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
) -> Tuple[YOCOForCausalLM, Dict]:
    llama = AutoModelForCausalLM.from_pretrained(
        llama_path,
        dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    llama_config = llama.config
    if not isinstance(llama_config, LlamaConfig):
        llama_config = LlamaConfig(**llama_config.to_dict())

    yoco_config = YOCOConfig(
        text_config=llama_config,
        num_self_decoder_layers=16,
        num_cross_decoder_layers=16,
        sliding_window=getattr(llama_config, "sliding_window", 1024) or 1024,
        pad_token_id=llama_config.pad_token_id,
        tie_word_embeddings=getattr(llama_config, "tie_word_embeddings", False),
    )

    yoco = YOCOForCausalLM(yoco_config)

    copy_embedding_and_heads(yoco, llama)
    copy_self_decoder(yoco, llama)
    copy_cross_decoder(yoco, llama)

    total_params, trainable_params = count_parameters(yoco)
    summary = {
        "llama_path": llama_path,
        "num_self_decoder_layers": yoco_config.num_self_decoder_layers,
        "num_cross_decoder_layers": yoco_config.num_cross_decoder_layers,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dtype": str(torch_dtype),
    }
    return yoco, summary


def save_summary(summary: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "init_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def validate_saved_checkpoint(output_dir: str, torch_dtype: torch.dtype) -> Dict:
    _, loading_info = YOCOForCausalLM.from_pretrained(
        output_dir,
        dtype=torch_dtype,
        output_loading_info=True,
    )
    return {
        "missing_keys": sorted(loading_info["missing_keys"]),
        "unexpected_keys": sorted(loading_info["unexpected_keys"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize YOCO-Llama from Llama")
    parser.add_argument(
        "--llama-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace path or local path to a Llama checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the initialized YOCO checkpoint",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Torch dtype used when loading the source Llama checkpoint",
    )
    parser.add_argument(
        "--runtime-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional writable directory for HuggingFace / torch runtime caches. "
            "If omitted, the script auto-selects a writable temp directory."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the Llama checkpoint only from local files / local HuggingFace cache.",
    )
    return parser.parse_args()


def _prepare_output_dir(output_dir: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create output directory: {output_dir}. "
            "Please choose a path you can write to, for example under /tmp or your workspace."
        ) from exc
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(
            f"Output directory is not writable: {output_dir}. "
            "Please choose a path you can write to."
        )


def _apply_runtime_cache_override(runtime_cache_dir: str) -> None:
    cache_env_map = {
        "HF_HOME": os.path.join(runtime_cache_dir, "hf_home"),
        "HF_HUB_CACHE": os.path.join(runtime_cache_dir, "hf_hub"),
        "TRANSFORMERS_CACHE": os.path.join(runtime_cache_dir, "transformers"),
        "XDG_CACHE_HOME": os.path.join(runtime_cache_dir, "xdg"),
        "TORCH_HOME": os.path.join(runtime_cache_dir, "torch"),
        "TORCH_EXTENSIONS_DIR": os.path.join(runtime_cache_dir, "torch_extensions"),
        "TRITON_CACHE_DIR": os.path.join(runtime_cache_dir, "triton"),
    }
    os.makedirs(runtime_cache_dir, exist_ok=True)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    for env_name, cache_dir in cache_env_map.items():
        os.makedirs(cache_dir, exist_ok=True)
        os.environ[env_name] = cache_dir


def main():
    args = parse_args()
    if args.runtime_cache_dir is not None:
        _apply_runtime_cache_override(args.runtime_cache_dir)
    _prepare_output_dir(args.output_dir)
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]
    try:
        yoco, summary = build_yoco_from_llama(
            args.llama_path,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        error_message = str(exc)
        if "xethub" in error_message or "CAS service error" in error_message or "dns error" in error_message:
            raise RuntimeError(
                "Failed to download the source Llama checkpoint through the HuggingFace Xet/CAS path. "
                "This usually means the current environment cannot resolve or reach transfer.xethub.hf.co. "
                "Use one of these workarounds:\n"
                "1. Pass a local checkpoint path to --llama-path.\n"
                "2. Re-run with --local-files-only after pre-downloading the model into your local HF cache.\n"
                "3. Ensure the current machine can access HuggingFace download endpoints.\n"
                f"Original error: {error_message}"
            ) from exc
        raise
    yoco.save_pretrained(args.output_dir)
    loading_info = validate_saved_checkpoint(args.output_dir, torch_dtype=torch_dtype)
    summary.update(loading_info)
    summary["runtime_cache_root"] = args.runtime_cache_dir or _RUNTIME_CACHE_ROOT
    save_summary(summary, args.output_dir)

    print("YOCO checkpoint initialized successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
