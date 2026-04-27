"""Prepare Phase 2 development subsets for the benchmark runner.

RULER is generated locally because it is a controlled synthetic stress test.
The other benchmarks are converted from their public HuggingFace datasets when
available. If a dataset is not cached and network access is unavailable, the
script reports the skipped source and continues with the datasets it can build.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scripts.schema import write_jsonl


FILLER_SENTENCE = (
    "This filler sentence is irrelevant to the answer and exists only to extend "
    "the context window for controlled long-context testing. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2 dev benchmark subsets")
    parser.add_argument("--output-root", default="benchmarks")
    parser.add_argument("--examples-per-source", type=int, default=2)
    parser.add_argument("--ruler-lengths", default="4096,8192,16384,32768")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def make_ruler_context(target_words: int, fact: str, position: str) -> str:
    filler_words = FILLER_SENTENCE.split()
    fact_words = fact.split()
    remaining = max(0, target_words - len(fact_words))
    left_words = remaining // 2
    if position == "early":
        left_words = max(32, remaining // 10)
    elif position == "late":
        left_words = max(32, remaining * 9 // 10)
    right_words = max(0, remaining - left_words)
    left = " ".join(filler_words[i % len(filler_words)] for i in range(left_words))
    right = " ".join(filler_words[i % len(filler_words)] for i in range(right_words))
    return f"{left}\n\n{fact}\n\n{right}"


def build_ruler(lengths: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = ["early", "middle", "late"]
    for length in lengths:
        for position in positions:
            passkey = f"RULER-{length}-{position}"
            fact = f"The secret passkey for this document is {passkey}."
            rows.append(
                {
                    "id": f"ruler_dev_{length}_{position}",
                    "benchmark": "RULER",
                    "task": f"passkey_retrieval_{position}",
                    "context": make_ruler_context(length, fact, position),
                    "question": "What is the secret passkey for this document?",
                    "answer": passkey,
                    "metadata": {
                        "target_words": length,
                        "needle_position": position,
                    },
                }
            )
    return rows


def try_load_dataset(repo: str, config: str, split: str, local_files_only: bool):
    try:
        from datasets import load_dataset

        return load_dataset(repo, config, split=split, download_mode="reuse_dataset_if_exists")
    except Exception as exc:
        if not local_files_only:
            print(f"Skipped {repo}/{config}: {type(exc).__name__}: {exc}")
            return None
        print(f"Skipped cached-only {repo}/{config}: {type(exc).__name__}: {exc}")
        return None


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value)


def select_rows(dataset, limit: int) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        rows.append(dict(row))
    return rows


def build_longbench(limit: int, local_files_only: bool) -> list[dict[str, Any]]:
    tasks = ["qasper_e", "hotpotqa_e", "gov_report_e", "lcc_e", "repobench-p_e"]
    output = []
    for task in tasks:
        dataset = try_load_dataset("THUDM/LongBench", task, "test", local_files_only)
        if dataset is None:
            continue
        for row in select_rows(dataset, limit):
            output.append(
                {
                    "id": f"longbench_dev_{task}_{row.get('_id', len(output))}",
                    "benchmark": "LongBench",
                    "task": str(row.get("dataset", task)),
                    "context": str(row.get("context", "")),
                    "question": str(row.get("input", "")),
                    "answer": normalize_answer(row.get("answers")),
                    "metadata": {
                        "source": "THUDM/LongBench",
                        "config": task,
                        "length": row.get("length"),
                        "language": row.get("language"),
                    },
                }
            )
    return output


def build_scbench(limit: int, local_files_only: bool) -> list[dict[str, Any]]:
    tasks = ["scbench_kv", "scbench_repoqa", "scbench_qa_eng", "scbench_summary"]
    output = []
    for task in tasks:
        dataset = try_load_dataset("microsoft/SCBench", task, "test", local_files_only)
        if dataset is None:
            continue
        for row in select_rows(dataset, limit):
            context = row.get("context") or row.get("input") or row.get("prompt") or ""
            multi_turns = row.get("multi_turns") or []
            first_turn = multi_turns[0] if multi_turns else {}
            question = (
                row.get("question")
                or row.get("query")
                or row.get("instruction")
                or first_turn.get("input")
                or first_turn.get("question")
                or ""
            )
            answer = (
                row.get("answer")
                or row.get("answers")
                or row.get("target")
                or row.get("label")
                or first_turn.get("answer")
            )
            output.append(
                {
                    "id": f"scbench_dev_{task}_{len(output)}",
                    "benchmark": "SCBench",
                    "task": task,
                    "context": str(context),
                    "question": str(question),
                    "answer": normalize_answer(answer),
                    "metadata": {
                        "source": "microsoft/SCBench",
                        "config": task,
                        "turn_index": 0 if multi_turns else None,
                        "raw_keys": sorted(row.keys()),
                    },
                }
            )
    return output


def build_longcodebench(limit: int, local_files_only: bool) -> list[dict[str, Any]]:
    output = []
    dataset = try_load_dataset("Steefano/LCB", "default", "test", local_files_only)
    rows: list[dict[str, Any]] = []
    if dataset is not None:
        rows = select_rows(dataset, limit)
    else:
        zip_path = Path(
            "/data3/junhaohu/.cache/huggingface/hub/"
            "datasets--Steefano--LCB/snapshots/"
            "989d5eff750d65a72c522e47f8f745ef2e22906b/LongCodeQA.zip"
        )
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                for bracket in ("32K", "64K", "128K"):
                    with archive.open(f"LQA/{bracket}.json") as f:
                        bracket_rows = json.load(f)
                    for row in bracket_rows[:limit]:
                        row = dict(row)
                        row["_context_bracket"] = bracket
                        rows.append(row)
        else:
            return output

    for row in rows[: max(limit, len(rows))]:
        context = row.get("context") or row.get("repo_text") or row.get("files") or row.get("prompt") or row.get("repo") or ""
        question = row.get("question") or row.get("issue") or row.get("instruction") or ""
        answer = row.get("answer") or row.get("gold") or row.get("target") or row.get("patch") or row.get("correct_letter")
        output.append(
            {
                "id": f"longcodebench_dev_{len(output)}",
                "benchmark": "LongCodeBench",
                "task": str(row.get("task", f"LongCodeQA_{row.get('_context_bracket', 'unknown')}")),
                "context": str(context),
                "question": str(question),
                "answer": normalize_answer(answer),
                "metadata": {
                    "source": "Steefano/LCB",
                    "context_bracket": row.get("_context_bracket"),
                    "raw_keys": sorted(row.keys()),
                },
            }
        )
    return output


def build_locomo(limit: int, local_files_only: bool) -> list[dict[str, Any]]:
    output = []
    dataset = try_load_dataset("Percena/locomo-mc10", "default", "train", local_files_only)
    rows: list[dict[str, Any]] = []
    if dataset is not None:
        rows = select_rows(dataset, limit)
    else:
        jsonl_path = Path(
            "/data3/junhaohu/.cache/huggingface/hub/"
            "datasets--Percena--locomo-mc10/snapshots/"
            "7d59a0463d83f97b042684310c0b3d17553004cd/data/locomo_mc10.json"
        )
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= limit:
                        break
                    rows.append(json.loads(line))
        else:
            return output

    for row in rows:
        context_parts = []
        for key in ("haystack_sessions", "conversation", "sessions", "context"):
            if row.get(key):
                context_parts.append(str(row[key]))
        context = "\n\n".join(context_parts)
        output.append(
            {
                "id": f"locomo_dev_{row.get('question_id', len(output))}",
                "benchmark": "LoCoMo",
                "task": str(row.get("question_type", "mc10")),
                "context": context,
                "question": str(row.get("question", "")),
                "answer": normalize_answer(row.get("answer")),
                "choices": row.get("choices"),
                "metadata": {
                    "source": "Percena/locomo-mc10",
                    "question_id": row.get("question_id"),
                    "raw_keys": sorted(row.keys()),
                },
            }
        )
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_root = Path(args.output_root)
    lengths = [int(item) for item in args.ruler_lengths.split(",") if item.strip()]

    builders = {
        "RULER": lambda: build_ruler(lengths),
        "LongBench": lambda: build_longbench(args.examples_per_source, args.local_files_only),
        "SCBench": lambda: build_scbench(args.examples_per_source, args.local_files_only),
        "LongCodeBench": lambda: build_longcodebench(args.examples_per_source, args.local_files_only),
        "LoCoMo": lambda: build_locomo(args.examples_per_source, args.local_files_only),
    }

    manifest = []
    for benchmark, builder in builders.items():
        rows = builder()
        output_path = output_root / benchmark / "dev.jsonl"
        write_jsonl(output_path, rows)
        manifest.append({"benchmark": benchmark, "path": str(output_path), "num_examples": len(rows)})
        print(f"Wrote {len(rows)} {benchmark} examples to {output_path}")

    manifest_path = output_root / "results" / "phase2_dev_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
