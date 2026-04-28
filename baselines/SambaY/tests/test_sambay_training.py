import csv
import subprocess
import sys
from pathlib import Path


def _run_training(tmp_path, total_steps, resume_ckpt=None):
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / f"model_step_{total_steps}"
    ckpt_dir = tmp_path / f"ckpt_step_{total_steps}"
    log_dir = tmp_path / "logs"
    cmd = [
        sys.executable,
        "baselines/SambaY/training/train_sambay_megatron.py",
        "--tiny-config",
        "--synthetic-data",
        "--device",
        "cpu",
        "--synthetic-num-samples",
        "8",
        "--synthetic-seq-len",
        "8",
        "--micro-batch-size",
        "2",
        "--total-steps",
        str(total_steps),
        "--lr",
        "0.01",
        "--warmup-steps",
        "1",
        "--log-interval",
        "1",
        "--steps-per-print",
        "1",
        "--output-dir",
        str(output_dir),
        "--ckpt-dir",
        str(ckpt_dir),
        "--log-dir",
        str(log_dir),
    ]
    if resume_ckpt is not None:
        cmd.extend(["--resume-ckpt", str(resume_ckpt)])

    result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=True)
    return result, output_dir, ckpt_dir, log_dir


def _read_losses(log_dir):
    with open(log_dir / "training_loss.csv", encoding="utf-8") as f:
        return [float(row["loss"]) for row in csv.DictReader(f, fieldnames=["dataset", "step", "loss"])]


def test_sambay_synthetic_training_loss_decreases_and_resumes(tmp_path):
    result, output_dir, ckpt_dir, log_dir = _run_training(tmp_path, total_steps=4)
    losses = _read_losses(log_dir)

    assert output_dir.joinpath("config.json").exists()
    assert ckpt_dir.joinpath("step_4", "model.pt").exists()
    assert len(losses) == 4
    assert losses[-1] < losses[0]
    assert "Trainable parameters:" in result.stdout

    resume_result, resume_output_dir, resume_ckpt_dir, _ = _run_training(
        tmp_path,
        total_steps=5,
        resume_ckpt=ckpt_dir / "step_4",
    )

    assert resume_output_dir.joinpath("config.json").exists()
    assert resume_ckpt_dir.joinpath("step_5", "model.pt").exists()
    assert "[synthetic] step 5" in resume_result.stdout
