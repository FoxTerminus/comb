import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import train_llama_repro as training


class ContextEvalOnlyTest(unittest.TestCase):
    def test_records_exact_loaded_step_without_optimizer(self):
        class Engine:
            module = object()

            def __init__(self):
                self.eval_calls = 0

            def eval(self):
                self.eval_calls += 1

        engine = Engine()
        calls = {}

        def fake_evaluate(model, dataset, examples, batch_size):
            calls.update(
                model=model,
                dataset=dataset,
                examples=examples,
                batch_size=batch_size,
            )
            return {"correct_context_nll": 0.25}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            training, "load_from_disk", return_value="dataset"
        ), patch.object(training, "rank0", return_value=True), patch.object(
            training.torch.distributed, "barrier"
        ), patch(
            "benchmarks.context_dependency_eval.evaluate_model",
            side_effect=fake_evaluate,
        ):
            output = Path(directory)
            training.evaluate_context_dependency(engine, output, 20_000, 64, 4)
            result = json.loads(
                (output / "context_dependency_step_00020000.json").read_text()
            )

        self.assertEqual(
            result, {"correct_context_nll": 0.25, "optimizer_step": 20_000}
        )
        self.assertEqual(
            calls,
            {
                "model": engine.module,
                "dataset": "dataset",
                "examples": 64,
                "batch_size": 4,
            },
        )
        self.assertEqual(engine.eval_calls, 1)

    def test_suite_uses_per_dataset_target_columns_and_writes_summary(self):
        class Engine:
            module = object()

            def eval(self):
                pass

        calls = []

        def fake_evaluate(
            model, dataset, examples, batch_size, target_column="teacher"
        ):
            calls.append((dataset, examples, batch_size, target_column))
            return {
                "examples": examples,
                "correct_context_nll": 0.2,
                "context_nll_gap": 0.5,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "suite.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "heldout",
                        "datasets": [
                            {
                                "name": "qa",
                                "path": str(root / "qa"),
                                "target_column": "labels",
                                "examples": 8,
                                "batch_size": 2,
                            },
                            {
                                "name": "summary",
                                "path": str(root / "summary"),
                                "target_column": "reference",
                                "examples": 4,
                            },
                        ],
                    }
                )
            )
            with patch.object(
                training,
                "load_from_disk",
                side_effect=lambda path: Path(path).name,
            ), patch.object(training, "rank0", return_value=True), patch.object(
                training.torch.distributed, "barrier"
            ), patch(
                "benchmarks.context_dependency_eval.evaluate_model",
                side_effect=fake_evaluate,
            ):
                training.evaluate_context_dependency(
                    Engine(), root, 25_000, 64, 4, str(manifest)
                )

            suite = json.loads(
                (root / "context_dependency_suite_step_00025000.json").read_text()
            )
            qa = json.loads(
                (root / "context_dependency_qa_step_00025000.json").read_text()
            )

        self.assertEqual(
            calls,
            [("qa", 8, 2, "labels"), ("summary", 4, 4, "reference")],
        )
        self.assertEqual(suite["suite_name"], "heldout")
        self.assertEqual(suite["datasets"]["qa"]["context_nll_gap"], 0.5)
        self.assertEqual(qa["optimizer_step"], 25_000)
        self.assertEqual(qa["target_column"], "labels")


if __name__ == "__main__":
    unittest.main()
