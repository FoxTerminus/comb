#!/usr/bin/env python3
"""Compare a training milestone's weight statistics with the official model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.artifact_io import write_text_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current = json.loads(args.current.read_text())
    reference = json.loads(args.reference.read_text())
    if current["trainable_parameters"] != reference["trainable_parameters"]:
        raise ValueError("trainable parameter counts differ")
    if set(current["groups"]) != set(reference["groups"]):
        raise ValueError("parameter groups differ")

    groups = {}
    for name in sorted(current["groups"]):
        this = current["groups"][name]
        other = reference["groups"][name]
        if this["parameters"] != other["parameters"]:
            raise ValueError(f"parameter count differs for {name}")
        groups[name] = {
            metric: {
                "current": this[metric],
                "reference": other[metric],
                "ratio": (
                    this[metric] / other[metric]
                    if float(other[metric]) != 0
                    else None
                ),
                "delta": this[metric] - other[metric],
            }
            for metric in (
                "root_mean_square",
                "standard_deviation",
                "mean_absolute",
                "max_absolute",
            )
        }

    gate_names = sorted(reference["gates"])
    if gate_names != sorted(current["gates"]):
        raise ValueError("gate names differ")
    current_gates = [float(current["gates"][name]["tanh"]) for name in gate_names]
    reference_gates = [
        float(reference["gates"][name]["tanh"]) for name in gate_names
    ]
    dot = sum(a * b for a, b in zip(current_gates, reference_gates, strict=True))
    current_norm = math.sqrt(sum(value * value for value in current_gates))
    reference_norm = math.sqrt(sum(value * value for value in reference_gates))
    gates = {
        "cosine_similarity": dot / (current_norm * reference_norm),
        "sign_agreement_fraction": sum(
            (a > 0) == (b > 0)
            for a, b in zip(current_gates, reference_gates, strict=True)
        )
        / len(gate_names),
        "current_l2": current_norm,
        "reference_l2": reference_norm,
        "l2_distance": math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(current_gates, reference_gates, strict=True)
            )
        ),
    }
    result = {
        "current": current["model"],
        "reference": reference["model"],
        "trainable_parameters": current["trainable_parameters"],
        "groups": groups,
        "gates": gates,
        "interpretation": (
            "Distributional similarity is a diagnostic, not proof that two "
            "models learned the same function."
        ),
    }
    write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
