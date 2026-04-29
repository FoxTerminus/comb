import pytest

from benchmarks.scripts.schema import BenchmarkExample, GenerationRecord, SchemaError
from benchmarks.scripts.utils.io import read_jsonl, write_jsonl


def test_example_round_trip(tmp_path):
    example = BenchmarkExample(
        id="x",
        benchmark="RULER",
        task="needle",
        split="smoke",
        context="alpha",
        question="what",
        answer="alpha",
        metadata={"context_length": 1, "source": "synthetic", "metric": "exact_match"},
    )
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [example.to_dict()])
    loaded = BenchmarkExample.from_dict(read_jsonl(path)[0])
    assert loaded == example


def test_generation_record_round_trip(tmp_path):
    record = GenerationRecord(
        run_id="run",
        model="mock",
        checkpoint="mock",
        benchmark="RULER",
        task="needle",
        id="x",
        prediction="alpha",
        answer="alpha",
        metrics={"primary_score": 1.0},
        example_metadata={"metric": "exact_match"},
        prompt_tokens=3,
        context_tokens=1,
        generated_tokens=1,
        kv_cache_policy="mock_no_kv_cache",
        chunk_size=None,
        recent_window_tokens=None,
        peak_memory_gb=None,
        prefill_latency_s=0.0,
        decode_latency_s=0.0,
        tokens_per_second=None,
        error=None,
    )
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [record.to_dict()])
    loaded = GenerationRecord.from_dict(read_jsonl(path)[0])
    assert loaded == record


def test_generation_record_rejects_unknown_kv_policy():
    row = GenerationRecord(
        run_id="run",
        model="mock",
        checkpoint="mock",
        benchmark="RULER",
        task="needle",
        id="x",
        prediction="alpha",
        answer="alpha",
        metrics={"primary_score": 1.0},
        example_metadata={"metric": "exact_match"},
        prompt_tokens=3,
        context_tokens=1,
        generated_tokens=1,
        kv_cache_policy="mock_no_kv_cache",
        chunk_size=None,
        recent_window_tokens=None,
        peak_memory_gb=None,
        prefill_latency_s=0.0,
        decode_latency_s=0.0,
        tokens_per_second=None,
        error=None,
    ).to_dict()
    row["kv_cache_policy"] = "made_up_policy"
    with pytest.raises(SchemaError, match="Unsupported kv_cache_policy"):
        GenerationRecord.from_dict(row)
