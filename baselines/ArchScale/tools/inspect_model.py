"""Inspect paper-aligned ArchScale baseline configs."""

from __future__ import annotations

import argparse
import json

from baselines.ArchScale.models.factory import (
    build_config,
    config_summary,
    describe_layer_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an ArchScale d=8 baseline")
    parser.add_argument(
        "--architecture",
        default="sambay",
        choices=["sambay", "sambayoco", "sambay_d8", "sambayoco_d8"],
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(
        args.architecture,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
    )
    payload = {
        "summary": config_summary(config),
        "layer_schedule": describe_layer_schedule(config),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    summary = payload["summary"]
    print("Config summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nLayer schedule")
    for item in payload["layer_schedule"]:
        flags = ", ".join(
            name
            for name in ["use_rnn", "use_gmu", "gmu_save", "yoco_kv", "yoco_cross"]
            if item[name]
        )
        suffix = f" ({flags})" if flags else ""
        print(f"  layer {item['layer']}: {item['mixer']}{suffix}")


if __name__ == "__main__":
    main()
