"""Bidirectional checkpoint converter between HuggingFace and Megatron TP formats.

Usage:
    # HuggingFace -> Megatron TP shards
    python models/checkpoint_converter.py hf2tp \
        --hf-path /path/to/hf_model \
        --output-dir /path/to/tp_shards \
        --tp-size 4

    # Megatron TP shards -> HuggingFace
    python models/checkpoint_converter.py tp2hf \
        --tp-dir /path/to/tp_checkpoint \
        --output-dir /path/to/hf_model \
        --tp-size 4
"""

import argparse
import os
import re
import torch
from collections import OrderedDict

from transformers import LlamaConfig
from models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

# Patterns for ColumnParallelLinear layers (split along dim=0 of weight)
COLUMN_PARALLEL_PATTERNS = [
    # Backbone self-attention
    r"language_model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.weight",
    # Backbone MLP
    r"language_model\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight",
    # Cross-attention q_proj
    r"language_model\.model\.cross_layers\.\d+\.cross_attn\.q_proj\.weight",
    # Cross-attention MLP
    r"language_model\.model\.cross_layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight",
    # Chunk encoder self-attention
    r"chunk_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)\.weight",
    # Chunk encoder MLP
    r"chunk_model\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight",
    # Chunk encoder k_proj / v_proj for cross-attention
    r"chunk_model\.(k_proj|v_proj)\.\d+\.weight",
    # LM head
    r"language_model\.lm_head\.weight",
]

# Patterns for RowParallelLinear layers (split along dim=1 of weight)
ROW_PARALLEL_PATTERNS = [
    # Backbone self-attention
    r"language_model\.model\.layers\.\d+\.self_attn\.o_proj\.weight",
    # Backbone MLP
    r"language_model\.model\.layers\.\d+\.mlp\.down_proj\.weight",
    # Cross-attention o_proj
    r"language_model\.model\.cross_layers\.\d+\.cross_attn\.o_proj\.weight",
    # Cross-attention MLP
    r"language_model\.model\.cross_layers\.\d+\.mlp\.down_proj\.weight",
    # Chunk encoder self-attention
    r"chunk_model\.layers\.\d+\.self_attn\.o_proj\.weight",
    # Chunk encoder MLP
    r"chunk_model\.layers\.\d+\.mlp\.down_proj\.weight",
]


def _is_column_parallel(key: str) -> bool:
    return any(re.match(p, key) for p in COLUMN_PARALLEL_PATTERNS)


def _is_row_parallel(key: str) -> bool:
    return any(re.match(p, key) for p in ROW_PARALLEL_PATTERNS)


# ---------------------------------------------------------------------------
# HuggingFace -> Megatron TP shards
# ---------------------------------------------------------------------------

def convert_hf_to_tp(hf_path: str, output_dir: str, tp_size: int):
    """Convert a HuggingFace CombLlama checkpoint to Megatron TP shards.

    Args:
        hf_path: Path to HuggingFace model (saved with save_pretrained).
        output_dir: Output directory for TP shard files.
        tp_size: Number of tensor parallel ranks.
    """
    print(f"Loading HuggingFace model from {hf_path}...")
    model = CombLlamaForConditionalGeneration.from_pretrained(hf_path)
    state_dict = model.state_dict()

    os.makedirs(output_dir, exist_ok=True)

    # Build per-rank state dicts
    shard_dicts = [OrderedDict() for _ in range(tp_size)]

    for key, tensor in state_dict.items():
        if _is_column_parallel(key):
            # Split along dim=0 (output features)
            chunks = torch.chunk(tensor, tp_size, dim=0)
            for rank, chunk in enumerate(chunks):
                shard_dicts[rank][key] = chunk.clone()
        elif _is_row_parallel(key):
            # Split along dim=1 (input features)
            chunks = torch.chunk(tensor, tp_size, dim=1)
            for rank, chunk in enumerate(chunks):
                shard_dicts[rank][key] = chunk.clone()
        else:
            # Replicated (embeddings, norms, gates, etc.)
            for rank in range(tp_size):
                shard_dicts[rank][key] = tensor.clone()

    # Save per-rank checkpoints
    for rank in range(tp_size):
        shard_path = os.path.join(output_dir, f"tp_rank_{rank}.pt")
        torch.save({
            "model_state_dict": shard_dicts[rank],
            "tp_size": tp_size,
        }, shard_path)
        print(f"  Saved TP rank {rank} -> {shard_path}")

    print(f"Done. {tp_size} TP shards saved to {output_dir}")


# ---------------------------------------------------------------------------
# Megatron TP shards -> HuggingFace
# ---------------------------------------------------------------------------

def convert_tp_to_hf(tp_dir: str, output_dir: str, tp_size: int,
                     model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Merge Megatron TP shard checkpoints back into a single HuggingFace model.

    Args:
        tp_dir: Directory containing tp_rank_0.pt, tp_rank_1.pt, etc.
        output_dir: Output directory for the merged HuggingFace model.
        tp_size: Number of tensor parallel ranks.
        model_name: Base model name for config.
    """
    # Load all TP shards
    shard_dicts = []
    for rank in range(tp_size):
        shard_path = os.path.join(tp_dir, f"tp_rank_{rank}.pt")
        print(f"  Loading TP rank {rank} from {shard_path}")
        ckpt = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_dicts.append(ckpt["model_state_dict"])

    # Merge into a single state dict
    merged = OrderedDict()
    all_keys = list(shard_dicts[0].keys())

    for key in all_keys:
        if _is_column_parallel(key):
            # Concatenate along dim=0
            merged[key] = torch.cat([sd[key] for sd in shard_dicts], dim=0)
        elif _is_row_parallel(key):
            # Concatenate along dim=1
            merged[key] = torch.cat([sd[key] for sd in shard_dicts], dim=1)
        else:
            # Replicated — take from rank 0
            merged[key] = shard_dicts[0][key]

    # Build model and load merged weights
    config = CombLlamaConfig(LlamaConfig.from_pretrained(model_name))
    model = CombLlamaForConditionalGeneration(config=config)
    model.load_state_dict(merged, strict=True)

    # Save as HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Merged HuggingFace model saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CombLlama Checkpoint Converter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # hf2tp
    p1 = subparsers.add_parser("hf2tp", help="Convert HuggingFace checkpoint to Megatron TP shards")
    p1.add_argument("--hf-path", required=True, help="Path to HuggingFace model directory")
    p1.add_argument("--output-dir", required=True, help="Output directory for TP shards")
    p1.add_argument("--tp-size", type=int, required=True, help="Tensor parallel size")

    # tp2hf
    p2 = subparsers.add_parser("tp2hf", help="Merge Megatron TP shards back to HuggingFace format")
    p2.add_argument("--tp-dir", required=True, help="Directory containing tp_rank_*.pt files")
    p2.add_argument("--output-dir", required=True, help="Output directory for merged HuggingFace model")
    p2.add_argument("--tp-size", type=int, required=True, help="Tensor parallel size")
    p2.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Base model name for config")

    args = parser.parse_args()

    if args.command == "hf2tp":
        convert_hf_to_tp(args.hf_path, args.output_dir, args.tp_size)
    elif args.command == "tp2hf":
        convert_tp_to_hf(args.tp_dir, args.output_dir, args.tp_size, args.model_name)


if __name__ == "__main__":
    main()
