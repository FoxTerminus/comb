#!/usr/bin/env python3
"""Print the per-layer topology of a SambaY / Samba+YOCO model.

Usage::

    python tools/inspect_model.py --config sambay_d16
    python tools/inspect_model.py --config sambayoco_d16
    python tools/inspect_model.py --config sambay_1b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mixer_name(blk) -> str:
    return type(blk.token_mixer).__name__


def _tags(blk) -> list[str]:
    """Collect the role tags for a single :class:`Block`."""
    tags: list[str] = []
    if blk.use_rnn:
        tags.append("RNN")
    if blk.use_gmu:
        tags.append("GMU")
    # KV_PRODUCER = yoco_kv set AND this layer actually produces K,V
    # (not cross-attention, which reads from the shared cache)
    is_kv_producer = blk.yoco_kv and not blk.yoco_cross and not blk.use_gmu
    if is_kv_producer:
        tags.append("KV_PRODUCER")
    if blk.yoco_cross:
        tags.append("CROSS_ATTN")
    if blk.use_full and not blk.yoco_kv and not blk.yoco_cross:
        tags.append("FULL_ATTN")
    if not tags:
        tags.append("LOCAL_SWA")
    return tags


def inspect(config_name: str) -> None:
    from baselines.ArchScale.models.config import Config
    from baselines.ArchScale.models.model import GPT

    if config_name.endswith(".yaml") or config_name.endswith(".yml"):
        cfg = Config.from_yaml(config_name)
    else:
        cfg = Config.from_name(config_name)

    model = GPT(cfg)
    total_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Config : {cfg.name}")
    print(f"n_embd : {cfg.n_embd}  |  n_head : {cfg.n_head}  |  "
          f"head_dim : {cfg.head_dim}  |  n_query_groups : {cfg.n_query_groups}")
    print(f"n_layer: {cfg.n_layer}  (self={cfg.num_self_decoder_layers}  "
          f"+ cross={cfg.num_cross_decoder_layers})")
    print(f"yoco   : {cfg.yoco}  |  gmu_yoco : {cfg.gmu_yoco}")
    print(f"params : {total_m:.1f}M")
    print()

    print(f"{'Layer':<6} {'Mixer':<28} {'Tags':<30} {'Self/Cross'}")
    print("-" * 78)

    mid = cfg.n_layer // 2
    for i, blk in enumerate(model.blocks):
        mixer = _mixer_name(blk)
        tags = " | ".join(_tags(blk))
        sec = "self_decoder" if i < mid else "cross_decoder"
        print(f"{i:<6} {mixer:<28} {tags:<30} {sec}")

    # ---- assertions --------------------------------------------------------
    gmu_count = sum(1 for b in model.blocks if b.use_gmu)
    kv_count = sum(1 for b in model.blocks if b.yoco_kv and not b.yoco_cross and not b.use_gmu)
    cross_count = sum(1 for b in model.blocks if b.yoco_cross)
    rnn_count = sum(1 for b in model.blocks if b.use_rnn)

    n = cfg.n_layer
    # expected values for n_layer=16
    exp_rnn = n // 2 // cfg.rnn_per_layer + 1  # self-decoder + boundary
    exp_kv = 1
    exp_cross = n // 2 - 2 + (0 if cfg.gmu_yoco else cfg.gmu_per_layer if cfg.gmu_per_layer > 0 else 0)

    print()
    print(f"GMU layers      : {gmu_count}  (expected: {'3 (gmu_yoco=True)' if cfg.gmu_yoco else '0 (gmu_yoco=False)'})")
    print(f"KV producer     : {kv_count}    (expected: 1)")
    print(f"RNN layers      : {rnn_count}")

    ok = True
    if cfg.gmu_yoco:
        if gmu_count != 3:
            print(f"  FAIL: expected 3 GMU layers")
            ok = False
    else:
        if gmu_count != 0:
            print(f"  FAIL: expected 0 GMU layers")
            ok = False

    if kv_count != 1:
        print(f"  FAIL: expected 1 KV producer")
        ok = False

    if ok:
        print("\nTopology verification PASSED")
    else:
        print("\nTopology verification FAILED")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Inspect SambaY/YOCO model topology")
    parser.add_argument(
        "--config", type=str, default="sambay_d16",
        help="Config name (e.g. sambay_d16) or YAML path",
    )
    args = parser.parse_args()
    inspect(args.config)


if __name__ == "__main__":
    main()
