"""Score benchmark predictions with task-aware diagnostic metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.metrics import (
    contains_match,
    edit_similarity,
    exact_match,
    normalize_text,
    rouge_l_f1,
    token_f1,
)
from benchmarks.scripts.schema import read_jsonl


BENCHMARKS = ["RULER", "LongBench", "SCBench", "LongCodeBench", "LoCoMo"]
LONGBENCH_QA_TASKS = {
    "narrativeqa",
    "qasper",
    "qasper_e",
    "multifieldqa_en",
    "multifieldqa_en_e",
    "multifieldqa_zh",
    "hotpotqa",
    "hotpotqa_e",
    "2wikimqa",
    "2wikimqa_e",
    "musique",
    "dureader",
    "triviaqa",
    "triviaqa_e",
}
LONGBENCH_SUMMARY_TASKS = {
    "gov_report",
    "gov_report_e",
    "qmsum",
    "multi_news",
    "multi_news_e",
    "vcsum",
    "samsum",
    "samsum_e",
}
LONGBENCH_CLASSIFICATION_TASKS = {"trec", "trec_e", "lsht"}
LONGBENCH_RETRIEVAL_TASKS = {
    "passage_count",
    "passage_count_e",
    "passage_retrieval_en",
    "passage_retrieval_en_e",
    "passage_retrieval_zh",
}
LONGBENCH_CODE_TASKS = {"lcc", "lcc_e", "repobench-p", "repobench-p_e"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score benchmark predictions")
    parser.add_argument("--manifest", required=True, help="Sweep manifest or a single predictions JSONL")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output-dir", default="benchmarks/reports/scored")
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for item in manifest:
            rows.extend(read_jsonl(item["path"]))
        return rows
    return read_jsonl(path)


def load_examples(split: str) -> dict[str, dict[str, Any]]:
    examples = {}
    for benchmark in BENCHMARKS:
        path = REPO_ROOT / "benchmarks" / benchmark / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            examples[row["id"]] = row
    return examples


def extract_choice_letter(text: Any) -> str | None:
    normalized = str(text or "").strip()
    match = re.search(r"\b([A-Z])\s*\)", normalized)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"\s*([A-Z])\b", normalized)
    if match:
        return match.group(1).upper()
    return None


def choice_score(prediction: str, answer: str | None, choices: list[str] | None) -> float | None:
    if answer is None:
        return None
    pred_letter = extract_choice_letter(prediction)
    answer_letter = answer.strip().upper() if len(answer.strip()) == 1 else None
    if pred_letter and answer_letter:
        return float(pred_letter == answer_letter)

    if choices:
        pred = normalize_text(prediction)
        answer_text = None
        if answer_letter and "A" <= answer_letter <= "Z":
            idx = ord(answer_letter) - ord("A")
            if idx < len(choices):
                answer_text = choices[idx]
        if answer_text is None:
            answer_text = answer
        return float(normalize_text(answer_text) in pred)

    return float(contains_match(prediction, answer) or False)


def longbench_score(task: str, prediction: str, answer: str | None) -> tuple[str, float | None]:
    if task in LONGBENCH_QA_TASKS:
        return "qa_f1", token_f1(prediction, answer)
    if task in LONGBENCH_SUMMARY_TASKS:
        return "rouge_l_f1", rouge_l_f1(prediction, answer)
    if task in LONGBENCH_CODE_TASKS:
        return "edit_similarity", edit_similarity(prediction, answer)
    if task in LONGBENCH_CLASSIFICATION_TASKS or task in LONGBENCH_RETRIEVAL_TASKS:
        return "exact_match", None if answer is None else float(exact_match(prediction, answer) or False)
    return "qa_f1", token_f1(prediction, answer)


def scbench_score(task: str, prediction: str, answer: str | None) -> tuple[str, float | None]:
    if task == "scbench_summary":
        return "rouge_l_f1", rouge_l_f1(prediction, answer)
    if task in {"scbench_repoqa", "scbench_repoqa_and_kv"}:
        return "edit_similarity", edit_similarity(prediction, answer)
    if task in {"scbench_choice_eng", "scbench_choice_chn"}:
        return "choice_or_exact", None if answer is None else float(contains_match(prediction, answer) or False)
    if task in {"scbench_kv", "scbench_prefix_suffix", "scbench_mf"}:
        return "exact_or_contains", None if answer is None else float(contains_match(prediction, answer) or False)
    return "qa_f1", token_f1(prediction, answer)


def score_row(row: dict[str, Any], example: dict[str, Any] | None) -> dict[str, Any]:
    if row.get("error"):
        return {
            **row,
            "metric_name": "error",
            "task_score": None,
            "scored_exact_match": None,
            "scored_contains_match": None,
            "scored_token_f1": None,
            "scored_rouge_l_f1": None,
            "scored_edit_similarity": None,
        }

    prediction = row.get("prediction")
    answer = row.get("answer")
    choices = example.get("choices") if example else None
    benchmark = row["benchmark"]
    task = row["task"]

    exact = exact_match(prediction, answer)
    contains = contains_match(prediction, answer)
    f1 = token_f1(prediction, answer)

    if benchmark == "RULER":
        task_score = None if contains is None else float(contains)
        metric_name = "passkey_contains"
    elif benchmark in {"LongCodeBench", "LoCoMo"} and (choices or len(str(answer or "").strip()) == 1):
        task_score = choice_score(str(prediction), None if answer is None else str(answer), choices)
        metric_name = "choice_match"
    elif benchmark == "SCBench":
        metric_name, task_score = scbench_score(task, str(prediction), None if answer is None else str(answer))
    elif benchmark == "LongBench":
        metric_name, task_score = longbench_score(task, str(prediction), None if answer is None else str(answer))
    else:
        task_score = None if contains is None else float(contains)
        metric_name = "contains_match"

    return {
        **row,
        "metric_name": metric_name,
        "task_score": task_score,
        "scored_exact_match": exact,
        "scored_contains_match": contains,
        "scored_token_f1": f1,
        "scored_rouge_l_f1": rouge_l_f1(prediction, answer),
        "scored_edit_similarity": edit_similarity(prediction, answer),
    }


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, float, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("retention_policy", "all_encoder_chunks")),
                float(row.get("compression_ratio", 1.0)),
                row["benchmark"],
                row["task"],
                row["metric_name"],
            )
        ].append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "compression_ratio",
            "retention_policy",
            "benchmark",
            "task",
            "metric_name",
            "num_examples",
            "score",
            "exact_match",
            "contains_match",
            "token_f1",
            "rouge_l_f1",
            "edit_similarity",
            "avg_peak_memory_gb",
            "avg_prefill_latency_s",
            "avg_decode_latency_s",
            "num_errors",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (retention_policy, ratio, benchmark, task, metric_name), items in sorted(groups.items()):
            writer.writerow(
                {
                    "compression_ratio": ratio,
                    "retention_policy": retention_policy,
                    "benchmark": benchmark,
                    "task": task,
                    "metric_name": metric_name,
                    "num_examples": len(items),
                    "score": mean([float(item["task_score"]) for item in items if item["task_score"] is not None]),
                    "exact_match": mean(
                        [float(item["scored_exact_match"]) for item in items if item["scored_exact_match"] is not None]
                    ),
                    "contains_match": mean(
                        [
                            float(item["scored_contains_match"])
                            for item in items
                            if item["scored_contains_match"] is not None
                        ]
                    ),
                    "token_f1": mean([item["scored_token_f1"] for item in items if item["scored_token_f1"] is not None]),
                    "rouge_l_f1": mean(
                        [item["scored_rouge_l_f1"] for item in items if item["scored_rouge_l_f1"] is not None]
                    ),
                    "edit_similarity": mean(
                        [
                            item["scored_edit_similarity"]
                            for item in items
                            if item["scored_edit_similarity"] is not None
                        ]
                    ),
                    "avg_peak_memory_gb": mean(
                        [float(item["peak_memory_gb"]) for item in items if item.get("peak_memory_gb") is not None]
                    ),
                    "avg_prefill_latency_s": mean([float(item["prefill_latency_s"]) for item in items]),
                    "avg_decode_latency_s": mean([float(item["decode_latency_s"]) for item in items]),
                    "num_errors": sum(1 for item in items if item.get("error")),
                }
            )


def write_overall(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("retention_policy", "all_encoder_chunks")),
                float(row.get("compression_ratio", 1.0)),
                row["benchmark"],
            )
        ].append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "compression_ratio",
            "retention_policy",
            "benchmark",
            "num_examples",
            "score",
            "avg_peak_memory_gb",
            "max_peak_memory_gb",
            "avg_prefill_latency_s",
            "avg_decode_latency_s",
            "num_errors",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (retention_policy, ratio, benchmark), items in sorted(groups.items()):
            memory = [float(item["peak_memory_gb"]) for item in items if item.get("peak_memory_gb") is not None]
            writer.writerow(
                {
                    "compression_ratio": ratio,
                    "retention_policy": retention_policy,
                    "benchmark": benchmark,
                    "num_examples": len(items),
                    "score": mean([float(item["task_score"]) for item in items if item["task_score"] is not None]),
                    "avg_peak_memory_gb": mean(memory),
                    "max_peak_memory_gb": max(memory) if memory else "",
                    "avg_prefill_latency_s": mean([float(item["prefill_latency_s"]) for item in items]),
                    "avg_decode_latency_s": mean([float(item["decode_latency_s"]) for item in items]),
                    "num_errors": sum(1 for item in items if item.get("error")),
                }
            )


def write_markdown_report(path: Path, benchmark_summary_path: Path, task_summary_path: Path) -> None:
    benchmark_rows = []
    with benchmark_summary_path.open("r", encoding="utf-8") as f:
        benchmark_rows = list(csv.DictReader(f))

    task_rows = []
    with task_summary_path.open("r", encoding="utf-8") as f:
        task_rows = list(csv.DictReader(f))

    lines = [
        "# Scored Benchmark Diagnostics",
        "",
        "These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.",
        "",
        "## Benchmark Summary",
        "",
        "| Policy | Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in benchmark_rows:
        lines.append(
            "| {retention_policy} | {compression_ratio} | {benchmark} | {num_examples} | {score:.3f} | {max_peak_memory_gb:.2f} | {avg_decode_latency_s:.3f} | {num_errors} |".format(
                retention_policy=row.get("retention_policy", "all_encoder_chunks"),
                compression_ratio=row["compression_ratio"],
                benchmark=row["benchmark"],
                num_examples=row["num_examples"],
                score=float(row["score"]) if row["score"] else 0.0,
                max_peak_memory_gb=float(row["max_peak_memory_gb"]) if row["max_peak_memory_gb"] else 0.0,
                avg_decode_latency_s=float(row["avg_decode_latency_s"]) if row["avg_decode_latency_s"] else 0.0,
                num_errors=row["num_errors"],
            )
        )

    lines.extend(
        [
            "",
            "## Task Summary",
            "",
        "| Policy | Compression | Benchmark | Task | Metric | Examples | Score |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in task_rows:
        lines.append(
            "| {retention_policy} | {compression_ratio} | {benchmark} | {task} | {metric_name} | {num_examples} | {score:.3f} |".format(
                retention_policy=row.get("retention_policy", "all_encoder_chunks"),
                compression_ratio=row["compression_ratio"],
                benchmark=row["benchmark"],
                task=row["task"],
                metric_name=row["metric_name"],
                num_examples=row["num_examples"],
                score=float(row["score"]) if row["score"] else 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "RULER uses passkey containment.",
            "",
            "LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.",
            "",
            "LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.",
            "",
            "SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.split)
    prediction_rows = load_prediction_rows(Path(args.manifest))
    scored_rows = [score_row(row, examples.get(row["id"])) for row in prediction_rows]

    write_jsonl(output_dir / "scored_predictions.jsonl", scored_rows)
    task_summary_path = output_dir / "task_summary.csv"
    benchmark_summary_path = output_dir / "benchmark_summary.csv"
    write_summary(task_summary_path, scored_rows)
    write_overall(benchmark_summary_path, scored_rows)
    write_markdown_report(output_dir / "score_report.md", benchmark_summary_path, task_summary_path)
    print(f"Wrote scored outputs to {output_dir}")


if __name__ == "__main__":
    main()
