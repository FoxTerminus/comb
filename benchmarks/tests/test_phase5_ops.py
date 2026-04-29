from benchmarks.scripts.reporting.failures import summarize_failures, write_failure_artifacts
from benchmarks.scripts.runners.retry import build_retry_manifest, write_retry_manifest
from benchmarks.scripts.runners.run_eval import run
from benchmarks.scripts.utils.config import load_run_config
from benchmarks.scripts.utils.environment import validate_environment_file, validate_environment_snapshot
from benchmarks.scripts.utils.io import read_json, read_jsonl


def test_failure_artifacts_and_retry_manifest(tmp_path):
    config = load_run_config("benchmarks/configs/runs/smoke_llama.json")
    config["model"]["model_path"] = str(tmp_path / "missing-model")
    config["model"]["tokenizer_path"] = str(tmp_path / "missing-tokenizer")
    output = run(config, output_dir=str(tmp_path / "failed"), run_id="pytest_failed", resume=False, limit=2)

    summary = summarize_failures(output)
    assert summary["failed_count"] == 2
    assert summary["by_error_type"] == {"ModelLoadError": 2}

    artifacts = write_failure_artifacts(output)
    assert artifacts["json"].exists()
    assert artifacts["csv"].exists()
    assert artifacts["markdown"].exists()

    retry = build_retry_manifest(output)
    assert retry["failed_count"] == 2
    assert retry["ids"] == [row["id"] for row in read_jsonl(output / "predictions.jsonl")]

    retry_path = write_retry_manifest(output)
    retry_json = read_json(retry_path)
    assert retry_json["ids"] == retry["ids"]


def test_retry_from_failed_run_filters_examples(tmp_path):
    failed_config = load_run_config("benchmarks/configs/runs/smoke_llama.json")
    failed_config["model"]["model_path"] = str(tmp_path / "missing-model")
    failed_config["model"]["tokenizer_path"] = str(tmp_path / "missing-tokenizer")
    failed_output = run(
        failed_config,
        output_dir=str(tmp_path / "failed"),
        run_id="pytest_failed",
        resume=False,
        limit=2,
    )

    mock_config = load_run_config("benchmarks/configs/runs/smoke_mock.json")
    retry_output = run(
        mock_config,
        output_dir=str(tmp_path / "retry"),
        run_id="pytest_retry",
        resume=False,
        limit=None,
        retry_from=str(failed_output),
    )
    rows = read_jsonl(retry_output / "predictions.jsonl")
    assert [row["id"] for row in rows] == build_retry_manifest(failed_output)["ids"]
    assert all(row["error"] is None for row in rows)

    retry_manifest = write_retry_manifest(failed_output)
    retry_from_manifest = run(
        mock_config,
        output_dir=str(tmp_path / "retry_manifest"),
        run_id="pytest_retry_manifest",
        resume=False,
        limit=None,
        only_ids_file=str(retry_manifest),
    )
    manifest_rows = read_jsonl(retry_from_manifest / "predictions.jsonl")
    assert [row["id"] for row in manifest_rows] == build_retry_manifest(failed_output)["ids"]


def test_environment_snapshot_validation(tmp_path):
    config = load_run_config("benchmarks/configs/runs/smoke_mock.json")
    output = run(config, output_dir=str(tmp_path / "env"), run_id="pytest_env", resume=False, limit=1)
    result = validate_environment_file(output / "environment.json")
    assert result["ok"] is True

    broken = read_json(output / "environment.json")
    del broken["git_commit"]
    problems = validate_environment_snapshot(broken)
    assert "missing key: git_commit" in problems
