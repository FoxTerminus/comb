"""Unified benchmark runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks.scripts.adapters.factory import build_adapter
from benchmarks.scripts.runners.retry import failure_ids
from benchmarks.scripts.reporting.report import (
    write_failure_csv,
    write_markdown_report,
    write_run_reports,
    write_summary_csv,
)
from benchmarks.scripts.schema import GenerationRecord
from benchmarks.scripts.scoring.score import attach_metrics
from benchmarks.scripts.utils.config import load_run_config, repo_path
from benchmarks.scripts.utils.data import load_examples
from benchmarks.scripts.utils.io import read_json, read_jsonl, write_json, write_jsonl
from benchmarks.scripts.utils.runtime import collect_environment, make_run_id, set_seed


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-context benchmark evaluation")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-ids-file", default=None)
    parser.add_argument("--retry-from", default=None)
    return parser.parse_args()


def _existing_successes(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    successes: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        if row.get("error") is None:
            successes[str(row.get("id"))] = row
    return successes


def _failure_record(
    *,
    run_id: str,
    model_config: dict[str, object],
    example,
    error: BaseException,
) -> GenerationRecord:
    return GenerationRecord(
        run_id=run_id,
        model=str(model_config.get("name", model_config.get("adapter", "unknown"))),
        checkpoint=str(model_config.get("model_path", "")),
        benchmark=example.benchmark,
        task=example.task,
        id=example.id,
        prediction="",
        answer=example.answer,
        metrics={},
        example_metadata=example.metadata,
        prompt_tokens=0,
        context_tokens=len(example.context.split()),
        generated_tokens=0,
        kv_cache_policy=str(model_config.get("kv_cache_policy", "unknown")),
        chunk_size=model_config.get("chunk_size"),  # type: ignore[arg-type]
        recent_window_tokens=model_config.get("recent_window_tokens"),  # type: ignore[arg-type]
        peak_memory_gb=None,
        prefill_latency_s=None,
        decode_latency_s=None,
        tokens_per_second=None,
        error=f"{type(error).__name__}: {error}",
    )


def _filter_summary_groups(summaries: list[dict[str, object]], prefixes: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        row
        for row in summaries
        if any(str(row.get("group", "")).startswith(prefix) for prefix in prefixes)
    ]


def _read_only_ids(path: str | None, retry_from: str | None) -> set[str] | None:
    if path and retry_from:
        raise ValueError("--only-ids-file and --retry-from are mutually exclusive")
    if retry_from:
        return set(failure_ids(retry_from))
    if not path:
        return None
    ids_path = repo_path(path)
    if ids_path.suffix == ".json":
        payload = read_json(ids_path)
        if not isinstance(payload.get("ids"), list):
            raise ValueError(f"JSON only-ids file must contain an ids list: {ids_path}")
        return {str(sample_id) for sample_id in payload["ids"]}
    rows = read_jsonl(ids_path)
    ids: set[str] = set()
    for row in rows:
        if "id" not in row:
            raise ValueError(f"Missing id in only-ids file row: {row}")
        ids.add(str(row["id"]))
    return ids


def run(
    config: dict[str, object],
    *,
    output_dir: str | None,
    run_id: str | None,
    resume: bool,
    limit: int | None,
    only_ids_file: str | None = None,
    retry_from: str | None = None,
) -> Path:
    seed = int(config.get("seed", 1234))
    set_seed(seed)
    run_name = str(config.get("run_name", "benchmark"))
    split = str(config.get("split", "smoke"))
    resolved_run_id = run_id or make_run_id(run_name)
    configured_output = config.get("paths", {}).get("output_dir") if isinstance(config.get("paths"), dict) else None
    default_output = f"benchmarks/results/{split}/{resolved_run_id}"
    output_path = repo_path(output_dir or (configured_output if run_id is None else None) or default_output)
    output_path.mkdir(parents=True, exist_ok=True)

    model_config = dict(config.get("model", {}))  # type: ignore[arg-type]
    generation_config = dict(config.get("generation", {}))  # type: ignore[arg-type]
    benchmark_names = list(config.get("benchmarks", []))  # type: ignore[arg-type]
    examples = load_examples(split, benchmark_names)
    only_ids = _read_only_ids(only_ids_file, retry_from)
    if only_ids is not None:
        examples = [example for example in examples if example.id in only_ids]
    if limit is not None:
        examples = examples[:limit]

    predictions_path = output_path / "predictions.jsonl"
    existing = _existing_successes(predictions_path) if resume else {}
    records: list[GenerationRecord] = [GenerationRecord.from_dict(row) for row in existing.values()]
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(output_path / "run_config.resolved.json", config)  # type: ignore[arg-type]

    adapter = build_adapter(model_config)
    for example in examples:
        if example.id in existing:
            continue
        try:
            record = adapter.generate(example, generation_config, resolved_run_id)
        except BaseException as exc:
            record = _failure_record(run_id=resolved_run_id, model_config=model_config, example=example, error=exc)
        record = attach_metrics(record, example)
        records.append(record)
        write_jsonl(predictions_path, [row.to_dict() for row in records])
        print(f"{record.model}/{example.benchmark}/{example.task}/{example.id}: error={record.error!r}")

    failures = [record.to_dict() for record in records if record.error]
    write_jsonl(output_path / "failures.jsonl", failures)
    end_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(
        output_path / "environment.json",
        collect_environment(model_config=model_config, start_time=start_time, end_time=end_time),
    )
    report_name = f"{split}_report"
    metrics = write_run_reports(output_path, records, report_name)

    formal_results_root = (REPO_ROOT / "benchmarks" / "results").resolve()
    if output_path.resolve().is_relative_to(formal_results_root):
        reports_dir = REPO_ROOT / "benchmarks" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / f"{resolved_run_id}_metrics.json", metrics)
        if split == "smoke":
            write_markdown_report(reports_dir / "smoke_report.md", title="Smoke Report", metrics=metrics)
        elif split == "dev":
            write_markdown_report(reports_dir / "dev_summary.md", title="Dev Summary", metrics=metrics)
        elif set(benchmark_names).issubset({"RULER", "SCBench", "LongBench"}):
            write_markdown_report(reports_dir / "primary_full_report.md", title="Primary Full Report", metrics=metrics)
        elif set(benchmark_names).issubset({"LoCoMo", "LongCodeBench"}):
            write_markdown_report(reports_dir / "secondary_full_report.md", title="Secondary Full Report", metrics=metrics)
        write_summary_csv(reports_dir / "benchmark_summary.csv", metrics["summaries"])
        write_summary_csv(
            reports_dir / "model_comparison.csv",
            _filter_summary_groups(metrics["summaries"], ("model/", "kv_cache_policy/", "overall")),
        )
        write_summary_csv(
            reports_dir / "metadata_summary.csv",
            _filter_summary_groups(metrics["summaries"], ("metadata/", "length_bucket/")),
        )
        write_failure_csv(reports_dir / "failure_summary.csv", records)
    return output_path


def main() -> None:
    args = parse_args()
    config = load_run_config(args.run_config)
    output_path = run(
        config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        resume=args.resume,
        limit=args.limit,
        only_ids_file=args.only_ids_file,
        retry_from=args.retry_from,
    )
    print(f"Wrote benchmark run to {output_path}")


if __name__ == "__main__":
    main()
