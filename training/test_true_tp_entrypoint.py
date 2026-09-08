from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
import tempfile
from pathlib import Path

from deepspeed.checkpoint.constants import UNIVERSAL_CHECKPOINT_INFO

from training import train_llama_true_tp_repro as entrypoint


class FakeEngine:
    def __init__(self):
        self.saved = None
        self.dp_world_size = 1

    def save_checkpoint(
        self,
        save_dir,
        tag=None,
        client_state=None,
        save_latest=True,
        exclude_frozen_parameters=False,
    ):
        self.saved = {
            "save_dir": save_dir,
            "tag": tag,
            "client_state": client_state,
            "save_latest": save_latest,
            "exclude_frozen_parameters": exclude_frozen_parameters,
        }
        return True


class TrueTPEntrypointTest(unittest.TestCase):
    def test_checkpoint_format_detection(self):
        self.assertFalse(entrypoint.should_load_universal_checkpoint(None, None))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest").write_text("step_native\n")
            self.assertFalse(
                entrypoint.should_load_universal_checkpoint(str(root), "step_native")
            )
            (root / "latest_universal").write_text("step_universal\n")
            self.assertTrue(
                entrypoint.should_load_universal_checkpoint(str(root), None)
            )
            marker = root / "step_universal" / "comb_universal_conversion_complete.json"
            marker.parent.mkdir()
            marker.write_text("{}\n")
            self.assertTrue(
                entrypoint.should_load_universal_checkpoint(
                    str(root), "step_universal"
                )
            )

    def test_initialization_and_checkpoint_metadata_are_isolated(self):
        engine = FakeEngine()
        captured = {}

        def fake_initialize(**kwargs):
            captured.update(kwargs)
            return engine, None, None, None

        model = object()
        bucket_calls = []

        class FakeDataset:
            def bucketing(self, local_rank, world_size):
                bucket_calls.append((local_rank, world_size))
                return ["same-bucket"]

        config = {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "tensor_parallel": {
                "autotp_size": 2,
                "tensor_parallel": {"tp_size": 2},
            },
        }
        original_initialize = entrypoint.base.deepspeed.initialize
        original_append_loss = entrypoint.base.append_loss
        try:
            entrypoint.base.deepspeed.initialize = fake_initialize
            with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as resume_root, patch.object(
                sys,
                "argv",
                [
                    "train_llama_true_tp_repro.py",
                    "--output-dir", output_dir,
                    "--resume-root", resume_root,
                    "--resume-tag", "step_source",
                ],
            ), patch.object(entrypoint.base, "DATASET_DICT", {"fake": FakeDataset}), patch.object(
                entrypoint, "apply_true_tp_comb_adapter",
                side_effect=lambda value, tp_size, dtype: value,
            ) as adapter, patch.object(
                entrypoint.base.torch.distributed, "get_rank", return_value=0
            ), patch.object(
                entrypoint.base.torch.cuda, "device_count", return_value=2
            ), patch.object(entrypoint.base.torch.cuda, "set_device") as set_device:
                marker = Path(resume_root) / "step_source" / "comb_universal_conversion_complete.json"
                marker.parent.mkdir()
                marker.write_text("{}\n")
                entrypoint.install_true_tp_hooks()
                result = entrypoint.base.deepspeed.initialize(
                    model=model, config=config
                )
                self.assertEqual(FakeDataset().bucketing(7, 99), ["same-bucket"])
                manifests = list(Path(output_dir).glob("true_tp_runtime_manifest_*.json"))
                self.assertEqual(len(manifests), 1)
                entrypoint.base.append_loss(
                    Path(output_dir) / "training_loss.csv",
                    [20_001, "SQuAD", 0, 1, 0.25, 5e-5],
                )
                timing = Path(output_dir) / "true_tp_step_timing.jsonl"
                self.assertIn('"optimizer_step": 20001', timing.read_text())
            self.assertIs(result[0], engine)
            adapter.assert_called_once()
            set_device.assert_called_once_with(0)
            self.assertIs(captured["model"], model)
            self.assertEqual(captured["config"]["train_batch_size"], 64)
            self.assertTrue(captured["config"]["checkpoint"]["load_universal"])
            self.assertEqual(captured["config"]["tensorboard"]["output_path"], output_dir)
            self.assertEqual(config["train_batch_size"], 32)
            self.assertEqual(bucket_calls, [(0, 1)])

            client_state = {"optimizer_step": 20_000}
            self.assertTrue(
                engine.save_checkpoint("/tmp/example", tag="step", client_state=client_state)
            )
            self.assertEqual(client_state, {"optimizer_step": 20_000})
            self.assertIn(UNIVERSAL_CHECKPOINT_INFO, engine.saved["client_state"])
        finally:
            entrypoint.base.deepspeed.initialize = original_initialize
            entrypoint.base.append_loss = original_append_loss


if __name__ == "__main__":
    unittest.main()
