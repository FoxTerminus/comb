import unittest

from data.SuperNI import split_superni_prompt
from data.curate_superni import exact_sample_size, task_seed


class CurateSuperNITest(unittest.TestCase):
    def test_default_prompt_split(self):
        instruction, context = split_superni_prompt("Classify the text.\nexample text")
        self.assertEqual(instruction, "Classify the text.")
        self.assertEqual(context, "example text")

    def test_single_line_prompt_is_safe(self):
        instruction, context = split_superni_prompt("Classify this")
        self.assertEqual(instruction, "Classify this")
        self.assertEqual(context, "")

    def test_exact_sample_size_keeps_small_tasks(self):
        self.assertEqual(exact_sample_size(1, 0.1), 1)
        self.assertEqual(exact_sample_size(15, 0.1), 2)
        self.assertEqual(exact_sample_size(100, 0.1), 10)

    def test_task_seed_is_stable_and_task_specific(self):
        self.assertEqual(task_seed("a", 42), task_seed("a", 42))
        self.assertNotEqual(task_seed("a", 42), task_seed("b", 42))


if __name__ == "__main__":
    unittest.main()
