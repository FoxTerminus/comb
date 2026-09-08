from __future__ import annotations

import unittest

from training.verify_hf_export_equivalence import loss_comparison


class HFExportEquivalenceTest(unittest.TestCase):
    def test_loss_comparison_accepts_only_finite_values_within_tolerance(self) -> None:
        self.assertTrue(loss_comparison([1.0, 1.0], 1.0005, 1e-3)["passed"])
        self.assertFalse(loss_comparison([1.0, 1.0], 1.002, 1e-3)["passed"])
        self.assertFalse(loss_comparison([1.0], float("nan"), 1e-3)["passed"])

    def test_loss_comparison_requires_rank_evidence(self) -> None:
        with self.assertRaises(ValueError):
            loss_comparison([], 1.0, 1e-3)


if __name__ == "__main__":
    unittest.main()
