import unittest
from types import SimpleNamespace

import torch

from benchmarks.validation_nll_eval import evaluate_model, make_batch


class FixedLossModel(torch.nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.losses = iter(losses)

    def forward(self, **_kwargs):
        return SimpleNamespace(loss=torch.tensor(next(self.losses)))


class ValidationBatchTest(unittest.TestCase):
    def test_shift_and_padding_match_training_semantics(self):
        rows = [
            {
                "input_ids": [10, 11, 12],
                "labels": [20, 21],
                "chunk_ids": [30, 31],
                "cross_attention_mask": [1, 1],
            },
            {
                "input_ids": [13, 14],
                "labels": [22, 23, 24],
                "chunk_ids": [32],
                "cross_attention_mask": [1],
            },
        ]
        batch = make_batch(rows, "cpu")
        self.assertEqual(batch["input_ids"].shape, (2, 512))
        self.assertEqual(batch["chunk_ids"].shape, (2, 2))
        self.assertEqual(batch["input_ids"][0, :5].tolist(), [10, 11, 12, 20, 21])
        self.assertEqual(batch["shift_labels"][0, :4].tolist(), [-100, -100, 20, 21])
        self.assertEqual(batch["shift_labels"][1, :4].tolist(), [-100, 22, 23, 24])
        self.assertEqual(batch["cross_attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["input_ids"].dtype, torch.int64)

    def test_validation_nll_is_weighted_by_target_tokens(self):
        dataset = [
            {
                "input_ids": [10, 11],
                "labels": [20],
                "chunk_ids": [30],
                "cross_attention_mask": [1],
            },
            {
                "input_ids": [12, 13],
                "labels": [21],
                "chunk_ids": [31],
                "cross_attention_mask": [1],
            },
            {
                "input_ids": [14, 15],
                "labels": [22, 23, 24],
                "chunk_ids": [32],
                "cross_attention_mask": [1],
            },
        ]
        result = evaluate_model(FixedLossModel([1.0, 3.0]), dataset, 2, "labels")
        self.assertEqual(result["target_tokens"], 5)
        self.assertAlmostEqual(result["validation_nll"], (1.0 * 2 + 3.0 * 3) / 5)

    def test_truncation_matches_training_shift_semantics(self):
        row = {
            "input_ids": list(range(511)),
            "labels": [600, 601, 602],
            "chunk_ids": [30],
            "cross_attention_mask": [1],
        }
        batch = make_batch([row], "cpu")
        positions = torch.nonzero(batch["shift_labels"][0] != -100).flatten()
        self.assertEqual(positions.tolist(), [510, 511])
        self.assertEqual(batch["shift_labels"][0, 510:].tolist(), [600, 601])


if __name__ == "__main__":
    unittest.main()
