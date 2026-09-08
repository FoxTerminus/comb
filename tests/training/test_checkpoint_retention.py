import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import train_llama_repro


class FakeEngine:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def save_checkpoint(self, save_dir, *, tag, client_state):
        if self.fail:
            raise RuntimeError("simulated checkpoint failure")
        root = Path(save_dir)
        (root / tag).mkdir()
        (root / "latest").write_text(tag + "\n")
        self.client_state = client_state


class CheckpointRetentionTest(unittest.TestCase):
    def test_success_keeps_only_two_newest_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2):
                (root / f"step_{step:08d}").mkdir()
            unrelated = root / "step_backup"
            unrelated.mkdir()
            engine = FakeEngine()

            with patch.object(train_llama_repro, "rank0", return_value=True), patch.object(
                train_llama_repro, "git_commit", return_value="test-commit"
            ), patch.object(train_llama_repro.torch.distributed, "barrier") as barrier:
                train_llama_repro.save_checkpoint(
                    engine,
                    root,
                    optimizer_step=3,
                    dataset_index=1,
                    bucket_index=2,
                    next_batch_index=4,
                    keep_last_checkpoints=2,
                )

            self.assertFalse((root / "step_00000001").exists())
            self.assertTrue((root / "step_00000002").is_dir())
            self.assertTrue((root / "step_00000003").is_dir())
            self.assertTrue(unrelated.is_dir())
            self.assertEqual((root / "latest").read_text().strip(), "step_00000003")
            self.assertEqual(
                json.loads((root / "latest_repro.json").read_text()),
                {"tag": "step_00000003", "optimizer_step": 3},
            )
            self.assertEqual(engine.client_state["next_batch_index"], 4)
            self.assertEqual(barrier.call_count, 2)

    def test_failed_save_does_not_remove_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2):
                (root / f"step_{step:08d}").mkdir()

            with self.assertRaisesRegex(RuntimeError, "simulated"), patch.object(
                train_llama_repro, "git_commit", return_value="test-commit"
            ), patch.object(train_llama_repro.torch.distributed, "barrier") as barrier:
                train_llama_repro.save_checkpoint(
                    FakeEngine(fail=True),
                    root,
                    optimizer_step=3,
                    dataset_index=1,
                    bucket_index=2,
                    next_batch_index=4,
                    keep_last_checkpoints=2,
                )

            self.assertTrue((root / "step_00000001").is_dir())
            self.assertTrue((root / "step_00000002").is_dir())
            self.assertFalse((root / "latest_repro.json").exists())
            barrier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
