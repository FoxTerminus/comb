import unittest

from data.curate_ni import deterministic_sample, exact_sample_size, task_seed


class CurateNITest(unittest.TestCase):
    def test_exact_sample_size_applies_fraction_and_cap(self):
        self.assertEqual(exact_sample_size(1000, 0.10, 512), 100)
        self.assertEqual(exact_sample_size(10000, 0.10, 512), 512)
        self.assertEqual(exact_sample_size(3, 0.10, 512), 1)
        self.assertEqual(exact_sample_size(0, 0.10, 512), 0)

    def test_sampling_is_deterministic_and_stratum_specific(self):
        values = list(range(100))
        first = deterministic_sample(
            values, count=10, task="task", stratum="short", seed=42
        )
        second = deterministic_sample(
            values, count=10, task="task", stratum="short", seed=42
        )
        medium = deterministic_sample(
            values, count=10, task="task", stratum="medium", seed=42
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, medium)
        self.assertNotEqual(task_seed("task", "short", 42), task_seed("task", "medium", 42))


if __name__ == "__main__":
    unittest.main()
