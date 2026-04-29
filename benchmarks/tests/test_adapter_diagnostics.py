from benchmarks.scripts.adapters.diagnostics import diagnose_adapter_config
from benchmarks.scripts.adapters.factory import build_adapter
from benchmarks.scripts.schema import BenchmarkExample


def test_mock_adapter_preflight_ok():
    diagnostics = diagnose_adapter_config({"adapter": "mock", "name": "mock", "model_path": "mock"})
    assert diagnostics.ok is True
    assert diagnostics.errors == []


def test_missing_local_model_path_is_reported_before_loading():
    diagnostics = diagnose_adapter_config(
        {
            "adapter": "llama",
            "name": "llama",
            "model_path": "/definitely/missing/model",
            "tokenizer_path": "/definitely/missing/tokenizer",
            "device": "cpu",
            "kv_cache_policy": "full_decoder_kv_cache",
        }
    )
    assert diagnostics.ok is False
    assert any("model_path does not exist" in error for error in diagnostics.errors)
    assert any("tokenizer_path does not exist" in error for error in diagnostics.errors)


def test_build_adapter_returns_load_error_adapter_with_preflight_message():
    adapter = build_adapter(
        {
            "adapter": "llama",
            "name": "llama",
            "model_path": "/definitely/missing/model",
            "tokenizer_path": "/definitely/missing/tokenizer",
            "device": "cpu",
            "kv_cache_policy": "full_decoder_kv_cache",
        }
    )
    example = BenchmarkExample(
        id="x",
        benchmark="RULER",
        task="needle_retrieval",
        split="smoke",
        context="The answer is ALPHA.",
        question="What is the answer?",
        answer="ALPHA",
        metadata={"metric": "exact_match", "source": "synthetic"},
    )
    record = adapter.generate(example, {"max_new_tokens": 1}, "diagnostic")
    assert "ModelLoadError" in str(record.error)
    assert "model_path does not exist" in str(record.error)
