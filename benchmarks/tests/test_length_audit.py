from benchmarks.scripts.auditing.length_audit import TokenCounter, audit_examples, length_bucket
from benchmarks.scripts.schema import BenchmarkExample


def _example(context: str) -> BenchmarkExample:
    return BenchmarkExample(
        id="x",
        benchmark="RULER",
        task="needle",
        split="dev",
        context=context,
        question="what is the answer",
        answer="answer",
        metadata={"context_length": len(context.split()), "source": "synthetic", "metric": "exact_match"},
    )


def test_length_bucket():
    assert length_bucket(100, [128, 256]) == "<= 128"
    assert length_bucket(300, [128, 256]) == "> 256"


def test_audit_examples_with_whitespace_counter():
    counter = TokenCounter(None)
    rows, summary = audit_examples(
        [_example("one two three four")],
        token_counter=counter,
        thresholds=[4, 8, 16],
        model_limits={
            "llama": {"max_context_tokens": 8, "kv_cache_policy": "full_decoder_kv_cache"},
            "combllama": {
                "max_context_tokens": 16,
                "kv_cache_policy": "chunk_encoder_cross_attention_kv",
                "chunk_size": 4,
                "recent_window_tokens": 3,
            },
        },
    )
    assert rows[0]["context_tokens"] == 4
    assert rows[0]["over_4"] is True
    assert "llama" in rows[0]["over_limit_models"]
    assert rows[0]["combllama_est_num_chunks"] > 0
    assert summary["record_count"] == 1
    assert summary["model_over_limit_counts"]["llama"] == 1


def test_missing_tokenizer_falls_back_to_whitespace(tmp_path):
    counter = TokenCounter(str(tmp_path / "missing-tokenizer"))
    assert counter.name == "whitespace"
    assert counter.error is not None
    assert counter.count_text("a b c") == 3

