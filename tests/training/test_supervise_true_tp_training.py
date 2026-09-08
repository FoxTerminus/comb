from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from training.supervise_true_tp_training import validate_baseline_if_needed


class ValidateBaselineIfNeededTest(unittest.TestCase):
    def test_native_resume_does_not_require_cleaned_historical_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "run"
            baseline = parent / "baseline"
            root.mkdir()
            baseline.mkdir()
            (root / "latest_repro.json").write_text("{}\n")
            validate_baseline_if_needed(root, baseline, "step_00020000", 20_000)

    def test_missing_native_resume_still_requires_complete_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "run"
            baseline = parent / "baseline"
            root.mkdir()
            baseline.mkdir()
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                validate_baseline_if_needed(
                    root, baseline, "step_00020000", 20_000
                )


if __name__ == "__main__":
    unittest.main()
