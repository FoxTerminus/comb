#!/usr/bin/env python
"""Score and aggregate LoCoMo shard outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR
from io_utils import read_jsonl, write_json
from score import score_record, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="full")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_name
    pred_dir = run_dir / "predictions"
    records = []
    for path in sorted(pred_dir.glob("*.jsonl")):
        records.extend(read_jsonl(path))
    scored = [score_record(r) for r in records if not r.get("error")]
    scored.sort(key=lambda r: (r["model"], int(r["global_index"])))
    summary = summarize(scored)

    write_json(run_dir / "scored_predictions.json", scored)
    write_json(run_dir / "scores.json", summary)
    _write_csv(run_dir / "scores.csv", summary)
    _write_markdown(run_dir / "scores.md", summary)

    conv_rows = []
    grouped = defaultdict(list)
    for rec in scored:
        grouped[(rec["model"], rec["sample_id"])].append(rec)
    for (model, sample_id), items in sorted(grouped.items()):
        conv_rows.append(
            {
                "model": model,
                "sample_id": sample_id,
                "n": len(items),
                "score": round(sum(float(x["score"]) for x in items) / len(items), 6),
            }
        )
    _write_csv(run_dir / "conversation_scores.csv", conv_rows)
    print(f"Wrote {run_dir / 'scores.csv'}")
    print(f"Wrote {run_dir / 'scores.md'}")


if __name__ == "__main__":
    main()

