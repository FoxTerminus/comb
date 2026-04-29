"""Initialize a SambaY-Llama checkpoint from a HuggingFace Llama checkpoint."""

import argparse
import json
import os
import tempfile
from typing import Dict, Tuple


def _ensure_writable_runtime_cache() -> str:
    preferred_roots = [
        "/data3/junhaohu/tmp",
        "/data3/junhaohu/tmp/build_tmp",
        "/custom_tmp",
        "/tmp",
        tempfile.gettempdir(),
    ]
    base_root = next((root for root in preferred_roots if os.path.isdir(root) and os.access(root, os.W_OK)), None)
    if base_root is None:
        base_root = tempfile.gettempdir()

    runtime_cache_root = os.path.join(base_root, "sambay_runtime_cache")
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

from baselines.SambaY.models.SambaY import SambaYConfig, SambaYForCausalLM, SambaYPureAttention


def count_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _copy_param(dst: torch.nn.Parameter, src: torch.Tensor):
    if dst.shape != src.shape:
        raise ValueError(f"Shape mismatch: dst={tuple(dst.shape)} src={tuple(src.shape)}")
    with torch.no_grad():
        dst.copy_(src)


def copy_embedding_and_heads(sambay: SambaYForCausalLM, llama) -> list[str]:
    copied = []
    _copy_param(sambay.model.embed_tokens.weight, llama.model.embed_tokens.weight.data)
    copied.append("model.embed_tokens.weight")
    _copy_param(sambay.model.norm.weight, llama.model.norm.weight.data)
    copied.append("model.norm.weight")
    _copy_param(sambay.lm_head.weight, llama.lm_head.weight.data)
    copied.append("lm_head.weight")
    return copied


def validate_llama_config_for_sambay(llama_config: LlamaConfig) -> None:
    if llama_config.num_hidden_layers % 4 != 0:
        raise ValueError(
            "ArchScale SambaY/YOCO schedule requires num_hidden_layers divisible by 4; "
            f"got {llama_config.num_hidden_layers}."
        )
    if llama_config.num_hidden_layers < 4:
        raise ValueError(f"SambaY requires at least 4 layers; got {llama_config.num_hidden_layers}.")
    if llama_config.hidden_size % llama_config.num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads: "
            f"{llama_config.hidden_size} % {llama_config.num_attention_heads} != 0."
        )
    if llama_config.num_attention_heads % llama_config.num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads for GQA: "
            f"{llama_config.num_attention_heads} % {llama_config.num_key_value_heads} != 0."
        )


def copy_mlp_and_norms(dst_layer, src_layer, prefix: str, copied: list[str]) -> None:
    _copy_param(dst_layer.input_layernorm.weight, src_layer.input_layernorm.weight.data)
    copied.append(f"{prefix}.input_layernorm.weight")
    _copy_param(dst_layer.post_attention_layernorm.weight, src_layer.post_attention_layernorm.weight.data)
    copied.append(f"{prefix}.post_attention_layernorm.weight")
    _copy_param(dst_layer.mlp.gate_proj.weight, src_layer.mlp.gate_proj.weight.data)
    copied.append(f"{prefix}.mlp.gate_proj.weight")
    _copy_param(dst_layer.mlp.up_proj.weight, src_layer.mlp.up_proj.weight.data)
    copied.append(f"{prefix}.mlp.up_proj.weight")
    _copy_param(dst_layer.mlp.down_proj.weight, src_layer.mlp.down_proj.weight.data)
    copied.append(f"{prefix}.mlp.down_proj.weight")


def copy_attention(dst_attn: SambaYPureAttention, src_attn, prefix: str, copied: list[str]) -> None:
    _copy_param(dst_attn.q_proj.weight, src_attn.q_proj.weight.data)
    copied.append(f"{prefix}.q_proj.weight")
    if hasattr(dst_attn, "k_proj"):
        _copy_param(dst_attn.k_proj.weight, src_attn.k_proj.weight.data)
        copied.append(f"{prefix}.k_proj.weight")
    if hasattr(dst_attn, "v_proj"):
        _copy_param(dst_attn.v_proj.weight, src_attn.v_proj.weight.data)
        copied.append(f"{prefix}.v_proj.weight")
    _copy_param(dst_attn.o_proj.weight, src_attn.o_proj.weight.data)
    copied.append(f"{prefix}.o_proj.weight")


def copy_self_decoder(sambay: SambaYForCausalLM, llama) -> list[str]:
    copied = []
    for idx, sambay_layer in enumerate(sambay.model.self_decoder.layers):
        llama_layer = llama.model.layers[idx]
        copy_mlp_and_norms(sambay_layer, llama_layer, f"model.self_decoder.layers.{idx}", copied)
        if isinstance(sambay_layer.token_mixer, SambaYPureAttention):
            copy_attention(
                sambay_layer.token_mixer,
                llama_layer.self_attn,
                f"model.self_decoder.layers.{idx}.token_mixer",
                copied,
            )

    gmu_save_idx = len(sambay.model.self_decoder.layers)
    gmu_save_source = llama.model.layers[gmu_save_idx]
    copy_mlp_and_norms(
        sambay.model.self_decoder.gmu_save_layer,
        gmu_save_source,
        "model.self_decoder.gmu_save_layer",
        copied,
    )

    boundary_idx = gmu_save_idx + 1
    boundary_source = llama.model.layers[boundary_idx]
    _copy_param(
        sambay.model.self_decoder.boundary_layer.input_layernorm.weight,
        boundary_source.input_layernorm.weight.data,
    )
    copied.append("model.self_decoder.boundary_layer.input_layernorm.weight")
    _copy_param(
        sambay.model.self_decoder.boundary_layer.post_attention_layernorm.weight,
        boundary_source.post_attention_layernorm.weight.data,
    )
    copied.append("model.self_decoder.boundary_layer.post_attention_layernorm.weight")
    _copy_param(
        sambay.model.self_decoder.boundary_layer.mlp.gate_proj.weight,
        boundary_source.mlp.gate_proj.weight.data,
    )
    copied.append("model.self_decoder.boundary_layer.mlp.gate_proj.weight")
    _copy_param(
        sambay.model.self_decoder.boundary_layer.mlp.up_proj.weight,
        boundary_source.mlp.up_proj.weight.data,
    )
    copied.append("model.self_decoder.boundary_layer.mlp.up_proj.weight")
    _copy_param(
        sambay.model.self_decoder.boundary_layer.mlp.down_proj.weight,
        boundary_source.mlp.down_proj.weight.data,
    )
    copied.append("model.self_decoder.boundary_layer.mlp.down_proj.weight")
    copy_attention(
        sambay.model.self_decoder.boundary_layer.self_attn,
        boundary_source.self_attn,
        "model.self_decoder.boundary_layer.self_attn",
        copied,
    )
    return copied


def copy_cross_decoder(sambay: SambaYForCausalLM, llama) -> list[str]:
    copied = []
    offset = sambay.config.num_self_decoder_layers + 2
    for idx, sambay_layer in enumerate(sambay.model.cross_decoder.layers):
        llama_layer = llama.model.layers[offset + idx]
        copy_mlp_and_norms(sambay_layer, llama_layer, f"model.cross_decoder.layers.{idx}", copied)
        if isinstance(sambay_layer.token_mixer, SambaYPureAttention):
            copy_attention(
                sambay_layer.token_mixer,
                llama_layer.self_attn,
                f"model.cross_decoder.layers.{idx}.token_mixer",
                copied,
            )
    return copied


def build_architecture_summary(sambay: SambaYForCausalLM, llama_config: LlamaConfig) -> Dict:
    self_layers = len(sambay.model.self_decoder.layers)
    cross_layers = len(sambay.model.cross_decoder.layers)
    total_arch_layers = self_layers + 1 + 1 + cross_layers
    expected_cross_layers = sambay.config.num_cross_decoder_layers - 2
    if total_arch_layers != llama_config.num_hidden_layers:
        raise RuntimeError(
            "SambaY module schedule does not match Llama layer count: "
            f"{total_arch_layers} != {llama_config.num_hidden_layers}."
        )
    if cross_layers != expected_cross_layers:
        raise RuntimeError(
            "Unexpected cross-decoder layer count: "
            f"{cross_layers} != {expected_cross_layers}."
        )

    cross_global_indices = [
        sambay.config.num_self_decoder_layers + 2 + idx
        for idx in range(cross_layers)
    ]
    return {
        "archscale_schedule": {
            "self_decoder_local_layers": [0, sambay.config.num_self_decoder_layers - 1],
            "gmu_save_layer": sambay.config.num_self_decoder_layers,
            "boundary_full_attention_layer": sambay.config.num_self_decoder_layers + 1,
            "cross_decoder_layers": [
                sambay.config.num_self_decoder_layers + 2,
                llama_config.num_hidden_layers - 1,
            ],
        },
        "actual_module_counts": {
            "self_decoder_local_layers": self_layers,
            "gmu_save_layers": 1,
            "boundary_full_attention_layers": 1,
            "cross_decoder_layers": cross_layers,
            "total_arch_layers": total_arch_layers,
        },
        "cross_decoder_global_layer_indices": cross_global_indices,
        "cross_decoder_uses_gmu": [layer.use_gmu for layer in sambay.model.cross_decoder.layers],
    }


def build_sambay_from_llama(
    llama_path: str,
    torch_dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
) -> Tuple[SambaYForCausalLM, Dict]:
    llama = AutoModelForCausalLM.from_pretrained(
        llama_path,
        dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    llama_config = llama.config
    if not isinstance(llama_config, LlamaConfig):
        llama_config = LlamaConfig(**llama_config.to_dict())
    validate_llama_config_for_sambay(llama_config)

    sambay_config = SambaYConfig(
        text_config=llama_config,
        num_self_decoder_layers=llama_config.num_hidden_layers // 2,
        num_cross_decoder_layers=llama_config.num_hidden_layers - llama_config.num_hidden_layers // 2,
        sliding_window=getattr(llama_config, "sliding_window", 1024) or 1024,
        pad_token_id=llama_config.pad_token_id,
        tie_word_embeddings=getattr(llama_config, "tie_word_embeddings", False),
    )
    sambay = SambaYForCausalLM(sambay_config)

    copied = []
    copied.extend(copy_embedding_and_heads(sambay, llama))
    copied.extend(copy_self_decoder(sambay, llama))
    copied.extend(copy_cross_decoder(sambay, llama))
    architecture_summary = build_architecture_summary(sambay, llama_config)

    total_params, trainable_params = count_parameters(sambay)
    summary = {
        "llama_path": llama_path,
        "num_self_decoder_layers": sambay_config.num_self_decoder_layers,
        "num_cross_decoder_layers": sambay_config.num_cross_decoder_layers,
        "rnn_per_layer": sambay_config.rnn_per_layer,
        "gmu_per_layer": sambay_config.gmu_per_layer,
        "gmu_memory_size": sambay_config.gmu_memory_size,
        "use_nope": sambay_config.use_nope,
        "llama_config": {
            "num_hidden_layers": llama_config.num_hidden_layers,
            "hidden_size": llama_config.hidden_size,
            "intermediate_size": llama_config.intermediate_size,
            "num_attention_heads": llama_config.num_attention_heads,
            "num_key_value_heads": llama_config.num_key_value_heads,
            "vocab_size": llama_config.vocab_size,
            "rms_norm_eps": llama_config.rms_norm_eps,
        },
        **architecture_summary,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dtype": str(torch_dtype),
        "copied_parameter_groups": copied,
        "newly_initialized_modules": [
            "Samba/Mamba token mixers, including the forced boundary gmu_save mixer",
            "GMU layers",
        ],
    }
    return sambay, summary


def save_summary(summary: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "init_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def validate_saved_checkpoint(output_dir: str, torch_dtype: torch.dtype) -> Dict:
    _, loading_info = SambaYForCausalLM.from_pretrained(
        output_dir,
        dtype=torch_dtype,
        output_loading_info=True,
    )
    return {
        "missing_keys": sorted(loading_info["missing_keys"]),
        "unexpected_keys": sorted(loading_info["unexpected_keys"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize SambaY-Llama from Llama")
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--runtime-cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")


def _apply_runtime_cache_override(runtime_cache_dir: str) -> None:
    os.makedirs(runtime_cache_dir, exist_ok=True)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    for env_name, suffix in {
        "HF_HOME": "hf_home",
        "HF_HUB_CACHE": "hf_hub",
        "TRANSFORMERS_CACHE": "transformers",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "TRITON_CACHE_DIR": "triton",
    }.items():
        cache_dir = os.path.join(runtime_cache_dir, suffix)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ[env_name] = cache_dir


def main():
    args = parse_args()
    if args.runtime_cache_dir is not None:
        _apply_runtime_cache_override(args.runtime_cache_dir)
    _prepare_output_dir(args.output_dir)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    sambay, summary = build_sambay_from_llama(
        args.llama_path,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    )
    sambay.save_pretrained(args.output_dir)
    summary.update(validate_saved_checkpoint(args.output_dir, torch_dtype=torch_dtype))
    summary["runtime_cache_root"] = args.runtime_cache_dir or _RUNTIME_CACHE_ROOT
    save_summary(summary, args.output_dir)

    print("SambaY checkpoint initialized successfully.")
    for key, value in summary.items():
        if key == "copied_parameter_groups":
            print(f"{key}: {len(value)} groups")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
