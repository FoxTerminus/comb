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
from typing import Dict, Tuple

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
        _copy_param(yoco_layer.self_attn.k_proj.weight, llama_layer.self_attn.k_proj.weight.data)
        _copy_param(yoco_layer.self_attn.v_proj.weight, llama_layer.self_attn.v_proj.weight.data)
        _copy_param(yoco_layer.self_attn.o_proj.weight, llama_layer.self_attn.o_proj.weight.data)

        _copy_param(yoco_layer.mlp.gate_proj.weight, llama_layer.mlp.gate_proj.weight.data)
        _copy_param(yoco_layer.mlp.up_proj.weight, llama_layer.mlp.up_proj.weight.data)
        _copy_param(yoco_layer.mlp.down_proj.weight, llama_layer.mlp.down_proj.weight.data)


def copy_cross_decoder(yoco: YOCOForCausalLM, llama) -> None:
    offset = yoco.config.num_self_decoder_layers
    for idx, yoco_layer in enumerate(yoco.model.cross_decoder.layers):
        llama_layer = llama.model.layers[offset + idx]

        _copy_param(yoco_layer.input_layernorm.weight, llama_layer.input_layernorm.weight.data)
        _copy_param(yoco_layer.post_attention_layernorm.weight, llama_layer.post_attention_layernorm.weight.data)

        # Use the original Llama self-attention projections as the initialization
        # source for the new cross-attention projections.
        _copy_param(yoco_layer.cross_attn.q_proj.weight, llama_layer.self_attn.q_proj.weight.data)
        _copy_param(yoco_layer.cross_attn.k_proj.weight, llama_layer.self_attn.k_proj.weight.data)
        _copy_param(yoco_layer.cross_attn.v_proj.weight, llama_layer.self_attn.v_proj.weight.data)
        _copy_param(yoco_layer.cross_attn.o_proj.weight, llama_layer.self_attn.o_proj.weight.data)

        _copy_param(yoco_layer.mlp.gate_proj.weight, llama_layer.mlp.gate_proj.weight.data)
        _copy_param(yoco_layer.mlp.up_proj.weight, llama_layer.mlp.up_proj.weight.data)
        _copy_param(yoco_layer.mlp.down_proj.weight, llama_layer.mlp.down_proj.weight.data)


def build_yoco_from_llama(
    llama_path: str,
    torch_dtype: torch.dtype = torch.float32,
) -> Tuple[YOCOForCausalLM, Dict]:
    llama = AutoModelForCausalLM.from_pretrained(llama_path, dtype=torch_dtype)
    llama_config = llama.config
    if not isinstance(llama_config, LlamaConfig):
        llama_config = LlamaConfig(**llama_config.to_dict())

    yoco_config = YOCOConfig(
        text_config=llama_config,
        num_self_decoder_layers=16,
        num_cross_decoder_layers=16,
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
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]
    yoco, summary = build_yoco_from_llama(args.llama_path, torch_dtype=torch_dtype)
    yoco.save_pretrained(args.output_dir)
    loading_info = validate_saved_checkpoint(args.output_dir, torch_dtype=torch_dtype)
    summary.update(loading_info)
    save_summary(summary, args.output_dir)

    print("YOCO checkpoint initialized successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
