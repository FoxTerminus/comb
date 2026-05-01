#!/usr/bin/env python3
"""Print Comb model topology and parameter counts.

Usage:
    python tools/inspect_comb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.config import CombConfig
from models.comb_qwen import CombForConditionalGeneration


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    return f"{n / 1e6:.1f}M"


def inspect():
    print("=== Comb + Qwen3-0.6B ===\n")

    config = CombConfig()
    print(f"cross_attention_layers: {config.cross_attention_layers}")
    print(f"num_cross_layers: {config.num_cross_layers}")
    print()

    print("Loading model (from_scratch=True, downloads Qwen3-0.6B)...")
    model = CombForConditionalGeneration(config, from_scratch=True)

    # Per-component counts
    decoder = sum(p.numel() for p in model.language_model.model.decoder_layers.parameters())
    embed = sum(p.numel() for p in model.language_model.model.embed_tokens.parameters())
    lm_head = sum(p.numel() for p in model.language_model.lm_head.parameters())
    dec_norm = sum(p.numel() for p in model.language_model.model.norm.parameters())

    enc_layers = sum(p.numel() for p in model.chunk_model.layers.parameters())
    enc_kv = sum(p.numel() for p in model.chunk_model.k_proj.parameters())
    enc_kv += sum(p.numel() for p in model.chunk_model.v_proj.parameters())
    enc_embed = sum(p.numel() for p in model.chunk_model.embed_tokens.parameters())

    cross = sum(p.numel() for p in model.language_model.model.cross_layers.parameters())

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print()
    print(f"  {'Component':<35} {'Params':>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'Decoder backbone (28 layers)':<35} {_fmt(decoder):>10}")
    print(f"  {'Decoder embedding':<35} {_fmt(embed):>10}")
    print(f"  {'Decoder LM head':<35} {_fmt(lm_head):>10}")
    print(f"  {'Decoder final norm':<35} {_fmt(dec_norm):>10}")
    print(f"  {'Chunk encoder layers (7)':<35} {_fmt(enc_layers):>10}")
    print(f"  {'Chunk encoder K/V proj (7×2)':<35} {_fmt(enc_kv):>10}")
    print(f"  {'Chunk encoder embedding':<35} {_fmt(enc_embed):>10}")
    print(f"  {'Cross-attention layers (7)':<35} {_fmt(cross):>10}")
    print(f"  {'  of which gates (14 params)':<35} {'<1M':>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'TOTAL':<35} {_fmt(total):>10}")
    print(f"  {'Trainable':<35} {_fmt(trainable):>10}")
    print(f"  {'Frozen':<35} {_fmt(frozen):>10}")
    print()

    # Verify freeze boundary
    frozen_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_params.append(name)
    print(f"Frozen param groups: {len(frozen_params)} tensors")

    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable param groups: {len(trainable_params)} tensors")

    # Check trainable is exactly encoder body + encoder K/V + cross-attn
    is_trainable_ok = all(
        any(prefix in name for prefix in [
            "chunk_model.layers", "chunk_model.k_proj", "chunk_model.v_proj",
            "cross_layers",
        ])
        for name in trainable_params
    )
    print(f"Trainable scope correct: {is_trainable_ok} ✓" if is_trainable_ok else "Trainable scope WRONG ✗")

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    inspect()
