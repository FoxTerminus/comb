#!/usr/bin/env python3
"""Validate the context-dependency diagnostic emitted after a training gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def validate_paired_units(
    payload: dict,
    *,
    units_key: str,
    index_key: str,
    count_key: str,
    mean_key: str,
    standard_error_key: str,
    positive_fraction_key: str,
) -> None:
    units = payload.get(units_key)
    if units is None:
        return
    if count_key in payload and len(units) != int(payload[count_key]):
        raise ValueError(f"Unexpected number of {units_key}")
    indices = [int(unit[index_key]) for unit in units]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"Invalid {units_key} indices")
    token_total = sum(int(unit["target_tokens"]) for unit in units)
    if token_total != int(payload["target_tokens"]):
        raise ValueError(f"{units_key} target-token count is inconsistent")

    gaps = []
    for unit in units:
        correct = float(unit["correct_context_nll"])
        shuffled = float(unit["shuffled_context_nll"])
        gap = float(unit["context_nll_gap"])
        if not all(math.isfinite(value) for value in (correct, shuffled, gap)):
            raise ValueError(f"Non-finite value in {units_key}")
        if not math.isclose(gap, shuffled - correct, abs_tol=1e-9):
            raise ValueError(f"Inconsistent gap in {units_key}")
        gaps.append(gap)

    mean = statistics.fmean(gaps)
    standard_error = statistics.stdev(gaps) / math.sqrt(len(gaps)) if len(gaps) > 1 else None
    positive_fraction = sum(gap > 0 for gap in gaps) / len(gaps)
    if not math.isclose(float(payload[mean_key]), mean, abs_tol=1e-12):
        raise ValueError(f"{mean_key} is inconsistent")
    reported_standard_error = payload[standard_error_key]
    if standard_error is None:
        if reported_standard_error is not None:
            raise ValueError(f"{standard_error_key} should be null")
    elif not math.isclose(
        float(reported_standard_error), standard_error, abs_tol=1e-12
    ):
        raise ValueError(f"{standard_error_key} is inconsistent")
    if not math.isclose(
        float(payload[positive_fraction_key]), positive_fraction, abs_tol=1e-12
    ):
        raise ValueError(f"{positive_fraction_key} is inconsistent")

    weighted_correct = sum(
        float(unit["correct_context_nll"]) * int(unit["target_tokens"])
        for unit in units
    ) / token_total
    weighted_shuffled = sum(
        float(unit["shuffled_context_nll"]) * int(unit["target_tokens"])
        for unit in units
    ) / token_total
    # Batch loss is accumulated in BF16 while per-example losses are reduced
    # separately; their token-weighted reconstructions can differ by a few
    # 1e-4 at NLLs above one.
    nll_abs_tol = 5e-4 if units_key == "paired_examples" else 1e-6
    if not math.isclose(
        weighted_correct, float(payload["correct_context_nll"]), abs_tol=nll_abs_tol
    ) or not math.isclose(
        weighted_shuffled, float(payload["shuffled_context_nll"]), abs_tol=nll_abs_tol
    ):
        raise ValueError(f"{units_key} weighted NLL is inconsistent")


def validate_distinct_units(payload: dict, units_key: str) -> None:
    units = payload[units_key]
    index_key = "start_example" if units_key == "distinct_batches" else "example_index"
    indices = [int(unit[index_key]) for unit in units]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"Invalid {units_key} indices")
    if units_key == "distinct_examples":
        if len(units) != int(payload["examples"]):
            raise ValueError("Unexpected number of distinct examples")
        for unit in units:
            if int(unit["wrong_context_example_index"]) == int(
                unit["example_index"]
            ):
                raise ValueError("A distinct-context pair reuses its own context")

    token_total = sum(int(unit["target_tokens"]) for unit in units)
    if token_total != int(payload["target_tokens"]):
        raise ValueError(f"{units_key} target-token count is inconsistent")
    gaps = []
    for unit in units:
        correct = float(unit["correct_context_nll"])
        wrong = float(unit["distinct_context_nll"])
        gap = float(unit["distinct_context_nll_gap"])
        if not all(math.isfinite(value) for value in (correct, wrong, gap)):
            raise ValueError(f"Non-finite value in {units_key}")
        if not math.isclose(gap, wrong - correct, abs_tol=1e-9):
            raise ValueError(f"Inconsistent gap in {units_key}")
        gaps.append(gap)

    prefix = "distinct_batch" if units_key == "distinct_batches" else "distinct_example"
    if not math.isclose(
        float(payload[f"{prefix}_gap_mean"]), statistics.fmean(gaps), abs_tol=1e-12
    ):
        raise ValueError(f"{prefix}_gap_mean is inconsistent")
    standard_error = (
        statistics.stdev(gaps) / math.sqrt(len(gaps)) if len(gaps) > 1 else None
    )
    reported = payload[f"{prefix}_gap_standard_error"]
    if standard_error is None:
        if reported is not None:
            raise ValueError(f"{prefix}_gap_standard_error should be null")
    elif not math.isclose(float(reported), standard_error, abs_tol=1e-12):
        raise ValueError(f"{prefix}_gap_standard_error is inconsistent")
    positive_key = f"distinct_positive_{'batch' if units_key == 'distinct_batches' else 'example'}_fraction"
    if not math.isclose(
        float(payload[positive_key]),
        sum(gap > 0 for gap in gaps) / len(gaps),
        abs_tol=1e-12,
    ):
        raise ValueError(f"{positive_key} is inconsistent")

    weighted_wrong = sum(
        float(unit["distinct_context_nll"]) * int(unit["target_tokens"])
        for unit in units
    ) / token_total
    tolerance = 5e-4 if units_key == "distinct_examples" else 1e-6
    if not math.isclose(
        weighted_wrong,
        float(payload["distinct_context_nll"]),
        abs_tol=tolerance,
    ):
        raise ValueError(f"{units_key} weighted distinct NLL is inconsistent")


def validate_no_context_units(payload: dict, units_key: str) -> None:
    units = payload[units_key]
    index_key = "start_example" if units_key == "no_context_batches" else "example_index"
    indices = [int(unit[index_key]) for unit in units]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"Invalid {units_key} indices")
    if units_key == "no_context_examples" and len(units) != int(payload["examples"]):
        raise ValueError("Unexpected number of no-context examples")
    token_total = sum(int(unit["target_tokens"]) for unit in units)
    if token_total != int(payload["target_tokens"]):
        raise ValueError(f"{units_key} target-token count is inconsistent")
    gaps = []
    for unit in units:
        correct = float(unit["correct_context_nll"])
        absent = float(unit["no_context_nll"])
        gap = float(unit["no_context_nll_gap"])
        if not all(math.isfinite(value) for value in (correct, absent, gap)):
            raise ValueError(f"Non-finite value in {units_key}")
        if not math.isclose(gap, absent - correct, abs_tol=1e-9):
            raise ValueError(f"Inconsistent gap in {units_key}")
        gaps.append(gap)
    weighted_absent = sum(
        float(unit["no_context_nll"]) * int(unit["target_tokens"])
        for unit in units
    ) / token_total
    tolerance = 5e-4 if units_key == "no_context_examples" else 1e-6
    if not math.isclose(
        weighted_absent, float(payload["no_context_nll"]), abs_tol=tolerance
    ):
        raise ValueError(f"{units_key} weighted no-context NLL is inconsistent")
    if units_key == "no_context_examples" and not math.isclose(
        float(payload["no_context_positive_example_fraction"]),
        sum(gap > 0 for gap in gaps) / len(gaps),
        abs_tol=1e-12,
    ):
        raise ValueError("no_context_positive_example_fraction is inconsistent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument(
        "--expected-gates",
        type=int,
        default=16,
        help="Expected number of context gates (16 for legacy Comb, 8 for Frozen32).",
    )
    args = parser.parse_args()

    payload = json.loads(args.result.read_text())
    if args.expected_step is not None:
        if int(payload["optimizer_step"]) != args.expected_step:
            raise ValueError("Unexpected optimizer step")
    if int(payload["examples"]) != args.expected_examples:
        raise ValueError("Unexpected example count")
    if int(payload["target_tokens"]) <= 0:
        raise ValueError("No target tokens were evaluated")

    numeric_keys = (
        "correct_context_nll",
        "shuffled_context_nll",
        "context_nll_gap",
        "distinct_context_nll",
        "distinct_context_nll_gap",
        "mean_abs_tanh_gate",
    )
    values = {key: float(payload[key]) for key in numeric_keys}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError({"nonfinite": values})
    expected_gap = values["shuffled_context_nll"] - values["correct_context_nll"]
    if not math.isclose(values["context_nll_gap"], expected_gap, abs_tol=1e-9):
        raise ValueError("context_nll_gap is inconsistent with the reported NLLs")
    expected_distinct_gap = (
        values["distinct_context_nll"] - values["correct_context_nll"]
    )
    if not math.isclose(
        values["distinct_context_nll_gap"], expected_distinct_gap, abs_tol=1e-9
    ):
        raise ValueError("distinct_context_nll_gap is inconsistent")

    no_context_keys = {
        "no_context_nll",
        "no_context_nll_gap",
        "no_context_positive_example_fraction",
        "no_context_batches",
        "no_context_examples",
    }
    present_no_context_keys = no_context_keys.intersection(payload)
    if present_no_context_keys and present_no_context_keys != no_context_keys:
        raise ValueError("Incomplete no-context diagnostic fields")
    if present_no_context_keys:
        no_context_nll = float(payload["no_context_nll"])
        no_context_gap = float(payload["no_context_nll_gap"])
        if not all(math.isfinite(value) for value in (no_context_nll, no_context_gap)):
            raise ValueError("Non-finite no-context diagnostic")
        if not math.isclose(
            no_context_gap,
            no_context_nll - values["correct_context_nll"],
            abs_tol=1e-9,
        ):
            raise ValueError("no_context_nll_gap is inconsistent")
        validate_no_context_units(payload, "no_context_batches")
        validate_no_context_units(payload, "no_context_examples")
        values["no_context_nll"] = no_context_nll
        values["no_context_nll_gap"] = no_context_gap

    gates = {key: float(value) for key, value in payload["gates"].items()}
    if len(gates) != args.expected_gates or not all(
        math.isfinite(value) for value in gates.values()
    ):
        raise ValueError({"invalid_gates": gates})
    mean_gate = sum(gates.values()) / len(gates)
    if not math.isclose(values["mean_abs_tanh_gate"], mean_gate, abs_tol=1e-9):
        raise ValueError("mean_abs_tanh_gate is inconsistent with gate values")

    validate_paired_units(
        payload,
        units_key="paired_batches",
        index_key="start_example",
        count_key="paired_batch_count",
        mean_key="paired_batch_gap_mean",
        standard_error_key="paired_batch_gap_standard_error",
        positive_fraction_key="positive_batch_fraction",
    ) if "paired_batches" in payload else None
    validate_paired_units(
        payload,
        units_key="paired_examples",
        index_key="example_index",
        count_key="examples",
        mean_key="paired_example_gap_mean",
        standard_error_key="paired_example_gap_standard_error",
        positive_fraction_key="positive_example_fraction",
    )
    validate_distinct_units(payload, "distinct_batches")
    validate_distinct_units(payload, "distinct_examples")

    print(json.dumps({"result": str(args.result.resolve()), "valid": True, **values}, indent=2))


if __name__ == "__main__":
    main()
