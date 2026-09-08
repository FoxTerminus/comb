from __future__ import annotations

import unittest

import torch

from benchmarks.audit_no_context_equivalence import comparison, tensor_sha256


class NoContextEquivalenceHelpersTest(unittest.TestCase):
    def test_exact_comparison(self) -> None:
        left = torch.tensor([[1.0, 2.0]])
        right = left.clone()
        report = comparison(left, right)
        self.assertTrue(report["exact_equal"])
        self.assertEqual(report["maximum_absolute_delta"], 0.0)
        self.assertEqual(report["left_sha256"], tensor_sha256(right))

    def test_signed_zero_has_canonical_fingerprint(self) -> None:
        positive = torch.tensor([0.0])
        negative = torch.tensor([-0.0])
        self.assertEqual(tensor_sha256(positive), tensor_sha256(negative))
        self.assertTrue(comparison(positive, negative)["exact_equal"])

    def test_numerically_equal_different_dtype_is_explicit(self) -> None:
        report = comparison(
            torch.tensor([1.0], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.float32),
        )
        self.assertTrue(report["exact_equal"])
        self.assertFalse(report["same_dtype"])
        self.assertEqual(report["left_sha256"], report["right_sha256"])

    def test_difference_and_shape_mismatch(self) -> None:
        report = comparison(torch.tensor([1.0]), torch.tensor([1.5]))
        self.assertFalse(report["exact_equal"])
        self.assertEqual(report["maximum_absolute_delta"], 0.5)
        mismatch = comparison(torch.ones(1), torch.ones(2))
        self.assertFalse(mismatch["same_shape"])
        self.assertIsNone(mismatch["maximum_absolute_delta"])


if __name__ == "__main__":
    unittest.main()
