import unittest

from benchmarks.context_dependency_eval import singleton_shuffled_context_rows


class SingletonShuffleTest(unittest.TestCase):
    def test_batch_one_uses_previous_global_context(self):
        rows = [{"id": 0}, {"id": 1}, {"id": 2}]
        self.assertEqual(singleton_shuffled_context_rows(rows, 0, 1), [rows[2]])
        self.assertEqual(singleton_shuffled_context_rows(rows, 2, 3), [rows[1]])

    def test_larger_batch_preserves_historical_in_batch_roll(self):
        rows = [{"id": 0}, {"id": 1}]
        self.assertIsNone(singleton_shuffled_context_rows(rows, 0, 2))


if __name__ == "__main__":
    unittest.main()
