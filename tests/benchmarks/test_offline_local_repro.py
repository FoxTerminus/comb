from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset

from benchmarks import offline_local_repro as local
from data.metrics import qa_f1_score


class FakeDatasetClass:
    metric = qa_f1_score

    @classmethod
    def scorer(cls, example, method):
        return {f"score_{method}": 0.5}


class LoadedDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)


class OfflineLocalReproductionTest(unittest.TestCase):
    def test_comb_uses_explicit_local_tokenizer_snapshot(self) -> None:
        data = Dataset.from_dict(
            {"input_ids": [[1]], "input_ids_new": [[2]], "labels": [[3]]}
        )
        first = SimpleNamespace(token_ids=[4], first_token_latency=2.0)
        second = SimpleNamespace(token_ids=[5], first_token_latency=0.5)
        fake_comb = mock.Mock()
        fake_comb.generate.side_effect = [[first], [second]]
        tokenizer = mock.Mock()
        tokenizer.decode.return_value = "prefixanswer"

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"COMB_TOKENIZER_PATH": directory}):
                with mock.patch.object(local, "COMB", return_value=fake_comb):
                    with mock.patch.object(
                        local.AutoTokenizer,
                        "from_pretrained",
                        return_value=tokenizer,
                    ) as from_pretrained:
                        result = local.run_comb("model", data, "prefix", 128)

        from_pretrained.assert_called_once_with(
            Path(directory), local_files_only=True
        )
        fake_comb.set_sampling_params.assert_called_once_with(
            temperature=0.0, max_tokens=128
        )
        self.assertEqual(result["output_comb"], ["answer"])
        self.assertEqual(result["ttft1_comb"], [2.0])
        self.assertEqual(result["ttft2_comb"], [0.5])

    def test_reuses_completed_normal_phase_after_comb_failure(self) -> None:
        base = Dataset.from_dict(
            {
                "input": ["q"] * 200,
                "context": ["c"] * 200,
                "answers": [["a"]] * 200,
                "normal_input": [[1]] * 200,
                "input_ids": [[2]] * 200,
                "normal_input_new": [[3]] * 200,
                "input_ids_new": [[4]] * 200,
                "chunk_ids": [[5]] * 200,
                "cross_attention_mask": [[1]] * 200,
                "labels": [[6]] * 200,
            }
        )

        def normal_phase(model, data, prefix, max_len):
            return data.add_column("output_normal", ["a"] * 200).add_column(
                "ttft1_normal", [2.0] * 200
            ).add_column("ttft2_normal", [1.0] * 200)

        def comb_phase(model, data, prefix, max_len):
            return data.add_column("output_comb", ["a"] * 200).add_column(
                "ttft1_comb", [1.0] * 200
            ).add_column("ttft2_comb", [0.25] * 200)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "hotpotqa.jsonl").write_text("{}\n")
            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                patches = (
                    mock.patch.object(local, "TEST_DATASETS", {"hotpotqa": FakeDatasetClass}),
                    mock.patch.object(local, "CHAT_TEMPLATE_PREFIX", {"model": ""}),
                    mock.patch.object(
                        local, "load_local_dataset", return_value=LoadedDataset(base)
                    ),
                    mock.patch.object(local, "run_vllm", side_effect=normal_phase),
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "offline_local_repro.py",
                            "--model",
                            "model",
                            "--dataset",
                            "hotpotqa",
                            "--data-dir",
                            str(data_dir),
                        ],
                    ),
                )
                with patches[0], patches[1], patches[2], patches[3] as run_vllm, patches[4]:
                    with mock.patch.object(local, "run_comb", side_effect=RuntimeError("boom")):
                        with self.assertRaisesRegex(RuntimeError, "boom"):
                            local.main()
                    checkpoint = Path(
                        "results/offline/model/.hotpotqa.normal.json"
                    )
                    self.assertTrue(checkpoint.is_file())
                    with mock.patch.object(local, "run_comb", side_effect=comb_phase):
                        local.main()
                    self.assertEqual(run_vllm.call_count, 1)
                    self.assertFalse(checkpoint.exists())
                    self.assertTrue(Path("results/offline/model/hotpotqa.json").is_file())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
