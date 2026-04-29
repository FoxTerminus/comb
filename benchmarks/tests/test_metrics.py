from benchmarks.scripts.scoring.metrics import (
    classification_accuracy,
    contains_match,
    edit_similarity,
    exact_match,
    rouge_l_f1,
    token_f1,
)
from benchmarks.scripts.schema import GenerationRecord
from benchmarks.scripts.schema import BenchmarkExample
from benchmarks.scripts.scoring.score import attach_metrics, summarize_records


def test_exact_and_contains():
    assert exact_match("The Answer", "answer") == 1.0
    assert contains_match("the secret is orchid", "orchid") == 1.0
    assert contains_match("", "orchid") == 0.0
    assert contains_match("or", "orchid") == 0.0


def test_f1_rouge_classification_and_code():
    assert token_f1("mira chen", "mira chen") == 1.0
    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert classification_accuracy("positive", "positive") == 1.0
    assert edit_similarity("def f(): pass", "def f(): pass") == 1.0


def test_summarize_records_groups_metadata_and_scores():
    record = GenerationRecord(
        run_id="run",
        model="mock",
        checkpoint="mock",
        benchmark="RULER",
        task="needle",
        id="x",
        prediction="alpha",
        answer="alpha",
        metrics={"primary_score": 1.0, "exact_match": 1.0, "contains_match": 1.0},
        example_metadata={"metric": "exact_match", "task_type": "retrieval", "source": "synthetic"},
        prompt_tokens=5000,
        context_tokens=10,
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
    summaries = summarize_records([record])["summaries"]
    groups = {row["group"]: row for row in summaries}
    assert groups["metadata/task_type/retrieval"]["primary_score"] == 1.0
    assert groups["length_bucket/<= 8192"]["max_prompt_tokens"] == 5000
    assert groups["overall"]["exact_match"] == 1.0


def test_failed_record_metrics_are_not_scored():
    example = BenchmarkExample(
        id="x",
        benchmark="RULER",
        task="needle",
        split="smoke",
        context="The answer is alpha.",
        question="What is the answer?",
        answer="alpha",
        metadata={"metric": "exact_match", "source": "synthetic"},
    )
    record = GenerationRecord(
        run_id="run",
        model="llama",
        checkpoint="missing",
        benchmark="RULER",
        task="needle",
        id="x",
        prediction="",
        answer="alpha",
        metrics={},
        example_metadata=example.metadata,
        prompt_tokens=10,
        context_tokens=4,
        generated_tokens=0,
        kv_cache_policy="full_decoder_kv_cache",
        chunk_size=None,
        recent_window_tokens=None,
        peak_memory_gb=None,
        prefill_latency_s=None,
        decode_latency_s=None,
        tokens_per_second=None,
        error="ModelLoadError: missing",
    )
    scored = attach_metrics(record, example)
    assert scored.metrics["primary_score"] is None
    assert summarize_records([scored])["summaries"][0]["success_count"] == 0
