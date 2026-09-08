import unittest

from datasets import Dataset

from training.train_llama_repro import slice_bucket_for_resume


class ResumeSliceTest(unittest.TestCase):
    def test_slice_preserves_remaining_batch_boundaries(self):
        dataset = Dataset.from_dict({"value": list(range(11))})
        sliced, offset = slice_bucket_for_resume(dataset, batch_size=4, skip_batches=2)

        self.assertEqual(offset, 2)
        self.assertEqual(sliced["value"], [8, 9, 10])
        self.assertEqual(
            [(relative + offset, sliced[start:start + 4]["value"])
             for relative, start in enumerate(range(0, len(sliced), 4))],
            [(2, [8, 9, 10])],
        )

    def test_zero_skip_returns_full_dataset(self):
        dataset = Dataset.from_dict({"value": [0, 1, 2]})
        sliced, offset = slice_bucket_for_resume(dataset, batch_size=2, skip_batches=0)
        self.assertIs(sliced, dataset)
        self.assertEqual(offset, 0)

    def test_rejects_checkpoint_past_bucket_end(self):
        dataset = Dataset.from_dict({"value": [0, 1, 2]})
        with self.assertRaisesRegex(ValueError, "bucket only contains"):
            slice_bucket_for_resume(dataset, batch_size=2, skip_batches=2)


if __name__ == "__main__":
    unittest.main()
