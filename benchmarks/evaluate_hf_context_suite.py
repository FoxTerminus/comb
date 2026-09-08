#!/usr/bin/env python3
"""Evaluate one local HF Comb checkpoint on a fixed PIC suite in one load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk

from benchmarks.context_dependency_eval import evaluate_model
from comb.integration.hf.CombLlama import CombLlamaForConditionalGeneration
from training.artifact_io import write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--panels",
        nargs="*",
        help="Optional panel names; by default evaluate every panel in the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    manifest_path = args.suite_manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads(manifest_path.read_text())
    requested = set(args.panels or [])
    panels = [
        panel
        for panel in manifest["datasets"]
        if not requested or panel["name"] in requested
    ]
    found = {panel["name"] for panel in panels}
    if requested - found:
        raise ValueError(f"unknown suite panels: {sorted(requested - found)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CombLlamaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda().eval()

    summaries = []
    for panel in panels:
        dataset = load_from_disk(panel["path"])
        result = evaluate_model(
            model,
            dataset,
            limit=int(panel["examples"]),
            batch_size=int(panel["batch_size"]),
            target_column=panel["target_column"],
        )
        record = {
            "model": str(model_path),
            "suite_manifest": str(manifest_path),
            "panel": panel,
            **result,
        }
        result_path = output_dir / f"context_dependency_{panel['name']}.json"
        write_text_atomic(result_path, json.dumps(record, indent=2) + "\n")
        summary = {"name": panel["name"]}
        summary.update(
            {
                key: result[key]
                for key in (
                "examples",
                "target_tokens",
                "correct_context_nll",
                "context_nll_gap",
                "positive_example_fraction",
                "distinct_context_nll_gap",
                "distinct_positive_example_fraction",
                "no_context_nll_gap",
                "no_context_positive_example_fraction",
                "mean_abs_tanh_gate",
                )
            }
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    write_text_atomic(
        output_dir / "suite_summary.json",
        json.dumps(
            {
                "model": str(model_path),
                "suite_manifest": str(manifest_path),
                "panels": summaries,
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
