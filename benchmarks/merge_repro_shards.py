"""Merge resumable benchmark shards with strict overlap validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from training.artifact_io import write_text_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--paper-target", type=float)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    merged: dict[int, dict] = {}
    for input_name in args.inputs:
        input_path = Path(input_name)
        with input_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["index"])
                if not args.start_index <= index < args.limit:
                    continue
                previous = merged.get(index)
                if previous is not None:
                    for key in ("dataset", "prediction", "score"):
                        if previous[key] != row[key]:
                            raise ValueError(
                                f"conflicting duplicate index {index} for {key}: "
                                f"{input_path}"
                            )
                else:
                    merged[index] = row

    expected = set(range(args.start_index, args.limit))
    actual = set(merged)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"incomplete shards: missing={missing}, extra={extra}")

    output_path = Path(args.output)
    rendered = "".join(
        json.dumps(merged[index], ensure_ascii=False) + "\n"
        for index in sorted(merged)
    )
    write_text_atomic(output_path, rendered)
    scores = [float(merged[index]["score"]) for index in sorted(merged)]
    summary = {
        "dataset": merged[args.start_index]["dataset"],
        "completed": len(merged),
        "mean_score": statistics.fmean(scores),
        "backends": sorted({row.get("backend", "unknown") for row in merged.values()}),
    }
    if args.paper_target is not None:
        summary["paper_target"] = args.paper_target
        summary["delta_from_paper"] = summary["mean_score"] - args.paper_target
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_text_atomic(summary_path, json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "rows": len(merged), **summary}))


if __name__ == "__main__":
    main()
