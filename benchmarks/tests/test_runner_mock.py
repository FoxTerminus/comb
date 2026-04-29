from benchmarks.scripts.runners.run_eval import run
from benchmarks.scripts.utils.config import load_run_config
from benchmarks.scripts.utils.io import read_jsonl


def test_mock_runner_generates_report(tmp_path):
    config = load_run_config("benchmarks/configs/runs/smoke_mock.json")
    output = run(config, output_dir=str(tmp_path / "run"), run_id="pytest", resume=False, limit=2)
    assert (output / "predictions.jsonl").exists()
    assert (output / "metrics.json").exists()
    assert (output / "smoke_report.md").exists()


def test_runner_resume_keeps_successful_predictions(tmp_path):
    config = load_run_config("benchmarks/configs/runs/smoke_mock.json")
    output = run(config, output_dir=str(tmp_path / "resume"), run_id="pytest", resume=False, limit=2)
    first_rows = read_jsonl(output / "predictions.jsonl")
    output = run(config, output_dir=str(tmp_path / "resume"), run_id="pytest", resume=True, limit=2)
    second_rows = read_jsonl(output / "predictions.jsonl")
    assert second_rows == first_rows


def test_model_load_failure_is_recorded_per_example(tmp_path):
    config = load_run_config("benchmarks/configs/runs/smoke_llama.json")
    config["model"]["model_path"] = str(tmp_path / "missing-model")
    config["model"]["tokenizer_path"] = str(tmp_path / "missing-tokenizer")
    output = run(config, output_dir=str(tmp_path / "failure"), run_id="pytest", resume=False, limit=2)
    rows = read_jsonl(output / "predictions.jsonl")
    assert len(rows) == 2
    assert all("ModelLoadError" in row["error"] for row in rows)
