import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from training.supervise_single_training import (
    committed_checkpoint,
    final_artifacts_complete,
    quarantine_uncommitted_checkpoints,
    rollback_loss_log,
    rotate_stdout,
)
from training.supervise_single_training import FINAL_STEP


FIELDS = ["optimizer_step", "dataset", "bucket", "batch", "loss", "lr"]


def write_checkpoint_archive(path: Path, payload: bytes = b"state") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload)


class SupervisorTest(unittest.TestCase):
    def test_final_artifacts_require_complete_indexed_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / f"hf_step_{FINAL_STEP:08d}"
            export.mkdir()
            (root / f"context_dependency_step_{FINAL_STEP:08d}.json").write_text(
                json.dumps({"optimizer_step": FINAL_STEP, "examples": 64})
            )
            (export / "config.json").write_text("{}")
            (export / "model-00001-of-00002.safetensors").write_bytes(b"first")
            (export / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "first.weight": "model-00001-of-00002.safetensors",
                            "second.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )
            self.assertFalse(
                final_artifacts_complete(root, run_strict_verifiers=False)
            )
            (export / "model-00002-of-00002.safetensors").write_bytes(b"second")
            self.assertTrue(
                final_artifacts_complete(root, run_strict_verifiers=False)
            )
            with patch(
                "training.supervise_single_training.subprocess.run"
            ) as verifier:
                self.assertTrue(final_artifacts_complete(root))
                self.assertEqual(verifier.call_count, 2)
                for call in verifier.call_args_list:
                    environment = call.kwargs["env"]
                    self.assertEqual(
                        environment["TRITON_CACHE_DIR"],
                        "/data3/junhaohu/.triton",
                    )
                    self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            (export / "stale.safetensors").write_bytes(b"stale")
            self.assertFalse(
                final_artifacts_complete(root, run_strict_verifiers=False)
            )
            (export / "stale.safetensors").unlink()
            (export / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "../escaped.safetensors"}})
            )
            self.assertFalse(
                final_artifacts_complete(root, run_strict_verifiers=False)
            )

    def test_final_artifacts_reject_wrong_context_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / f"hf_step_{FINAL_STEP:08d}"
            export.mkdir()
            context = root / f"context_dependency_step_{FINAL_STEP:08d}.json"
            context.write_text(
                json.dumps({"optimizer_step": FINAL_STEP - 1, "examples": 64})
            )
            (export / "config.json").write_text("{}")
            (export / "model.safetensors").write_bytes(b"weights")
            (export / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model.safetensors"}})
            )
            self.assertFalse(
                final_artifacts_complete(root, run_strict_verifiers=False)
            )
            context.write_text(
                json.dumps({"optimizer_step": FINAL_STEP, "examples": 64})
            )
            (export / "unfinished.tmp").write_bytes(b"partial")
            self.assertFalse(
                final_artifacts_complete(root, run_strict_verifiers=False)
            )

    def test_final_artifacts_reject_failed_strict_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / f"hf_step_{FINAL_STEP:08d}"
            export.mkdir()
            (root / f"context_dependency_step_{FINAL_STEP:08d}.json").write_text(
                json.dumps({"optimizer_step": FINAL_STEP, "examples": 64})
            )
            (export / "config.json").write_text("{}")
            (export / "model.safetensors").write_bytes(b"weights")
            (export / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model.safetensors"}})
            )
            with patch(
                "training.supervise_single_training.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "verifier"),
            ):
                self.assertFalse(final_artifacts_complete(root))

    def test_committed_checkpoint_requires_consistent_complete_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tag = "step_00000002"
            checkpoint = root / tag
            checkpoint.mkdir()
            write_checkpoint_archive(checkpoint / "mp_rank_00_model_states.pt")
            write_checkpoint_archive(
                checkpoint / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
            )
            (root / "latest").write_text(tag)
            (root / "latest_repro.json").write_text(
                json.dumps({"tag": tag, "optimizer_step": 2})
            )
            self.assertEqual(committed_checkpoint(root), (tag, 2))
            (root / "latest").write_text("step_00000001")
            with self.assertRaises(RuntimeError):
                committed_checkpoint(root)

    def test_committed_checkpoint_validates_all_true_tp_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tag = "step_00021000"
            checkpoint = root / tag
            checkpoint.mkdir()
            for rank in range(2):
                write_checkpoint_archive(
                    checkpoint / f"mp_rank_{rank:02d}_model_states.pt"
                )
                write_checkpoint_archive(
                    checkpoint
                    / f"bf16_zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt"
                )
            (root / "latest").write_text(tag)
            (root / "latest_repro.json").write_text(
                json.dumps({"tag": tag, "optimizer_step": 21000})
            )
            self.assertEqual(committed_checkpoint(root, 2), (tag, 21000))
            (checkpoint / "mp_rank_01_model_states.pt").unlink()
            with self.assertRaises(RuntimeError):
                committed_checkpoint(root, 2)

    def test_committed_checkpoint_falls_back_to_verified_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tag = "step_00021000"
            checkpoint = root / tag
            checkpoint.mkdir()
            files = {}
            for rank in range(2):
                for name in (
                    f"mp_rank_{rank:02d}_model_states.pt",
                    f"bf16_zero_pp_rank_0_mp_rank_{rank:02d}_optim_states.pt",
                ):
                    path = checkpoint / name
                    write_checkpoint_archive(path)
                    files[name] = path.stat().st_size
            (root / "latest").write_text("step_00022000")
            (root / "latest_repro.json").write_text("{truncated")
            (root / "last_verified_checkpoint.json").write_text(
                json.dumps(
                    {"tag": tag, "optimizer_step": 21000, "files": files}
                )
            )
            self.assertEqual(committed_checkpoint(root, 2), (tag, 21000))
            (checkpoint / "mp_rank_01_model_states.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "no recoverable"):
                committed_checkpoint(root, 2)

    def test_loss_rollback_is_exact_and_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "training_loss.csv"
            rows = [
                {
                    "optimizer_step": step,
                    "dataset": "test",
                    "bucket": 0,
                    "batch": step,
                    "loss": step / 10,
                    "lr": 0.1,
                }
                for step in range(1, 6)
            ]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            archive = rollback_loss_log(root, 3, "test")
            self.assertIsNotNone(archive)
            with path.open(newline="") as handle:
                kept = list(csv.DictReader(handle))
            with archive.open(newline="") as handle:
                dropped = list(csv.DictReader(handle))
            self.assertEqual([int(row["optimizer_step"]) for row in kept], [1, 2, 3])
            self.assertEqual([int(row["optimizer_step"]) for row in dropped], [4, 5])
            self.assertIsNone(rollback_loss_log(root, 3, "unused"))

    def test_loss_rollback_refuses_noncontiguous_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "training_loss.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(
                    [
                        dict.fromkeys(FIELDS, 1),
                        {**dict.fromkeys(FIELDS, 3), "optimizer_step": 3},
                    ]
                )
            with self.assertRaises(RuntimeError):
                rollback_loss_log(root, 1, "test")

    def test_loss_rollback_supports_resumed_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "training_loss.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                for step in range(20001, 20006):
                    writer.writerow([step, "test", 0, step, 1.0, 0.1])
            archive = rollback_loss_log(
                root, 20003, "offset", expected_first_step=20001
            )
            self.assertIsNotNone(archive)
            with path.open(newline="") as handle:
                steps = [int(row["optimizer_step"]) for row in csv.DictReader(handle)]
            self.assertEqual(steps, [20001, 20002, 20003])

    def test_uncommitted_checkpoint_and_stdout_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "run"
            root.mkdir()
            committed = root / "step_00000010"
            incomplete = root / "step_00000011"
            committed.mkdir()
            incomplete.mkdir()
            (incomplete / "partial").write_bytes(b"recoverable")
            stdout = parent / "run.stdout.log"
            stdout.write_text("failure evidence\n")
            quarantined = quarantine_uncommitted_checkpoints(root, 10, "test")
            self.assertEqual(len(quarantined), 1)
            self.assertEqual((quarantined[0] / "partial").read_bytes(), b"recoverable")
            self.assertTrue(committed.is_dir())
            prior_stdout = rotate_stdout(root, "test")
            self.assertEqual(prior_stdout.read_text(), "failure evidence\n")
            self.assertFalse(stdout.exists())


if __name__ == "__main__":
    unittest.main()
