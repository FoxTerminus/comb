from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from training.compare_hf_checkpoint_weights import compare


def make_checkpoint(root: Path, value: float) -> None:
    root.mkdir()
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {"weight": "model.safetensors"},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    save_file({"weight": torch.tensor([value, 2.0])}, root / "model.safetensors")


class CompareHfCheckpointWeightsTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root / "left", 1.0)
            make_checkpoint(root / "right", 1.0)
            report = compare(root / "left", root / "right")
            self.assertTrue(report["passed"])
            self.assertEqual(report["elements"], 2)

    def test_value_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root / "left", 1.0)
            make_checkpoint(root / "right", 3.0)
            report = compare(root / "left", root / "right")
            self.assertFalse(report["passed"])
            self.assertEqual(report["unequal_tensors"], ["weight"])


if __name__ == "__main__":
    unittest.main()
