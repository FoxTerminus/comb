from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from deepspeed.checkpoint.constants import UNIVERSAL_CHECKPOINT_INFO
from safetensors import safe_open
import torch

from training.export_true_tp_checkpoint_to_hf_cpu import export_checkpoint


class CpuTrueTpExporterTest(unittest.TestCase):
    def test_merges_replicated_column_and_row_parallel_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            template = root / "template"
            output = root / "output"
            checkpoint.mkdir()
            template.mkdir()
            embed_name = "chunk_model.embed_tokens.weight"
            names = ["norm.weight", "q_proj.weight", "o_proj.weight", embed_name]
            weight_map = {name: "model-00001-of-00001.safetensors" for name in names}
            expected = {
                "norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
                "q_proj.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
                "o_proj.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
                embed_name: torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            }
            index = {
                "metadata": {"total_size": sum(x.numel() * 2 for x in expected.values())},
                "weight_map": weight_map,
            }
            (template / "model.safetensors.index.json").write_text(json.dumps(index))
            (template / "config.json").write_text("{}\n")
            info = {
                "tp_replicated_parameter_patterns": [r"norm\.weight"],
                "parameter_with_row_parallelism_patterns": [r"o_proj\.weight"],
            }
            for rank in range(2):
                module = {
                    "norm.weight": expected["norm.weight"].clone(),
                    "q_proj.weight": expected["q_proj.weight"][rank * 2 : (rank + 1) * 2].clone(),
                    "o_proj.weight": expected["o_proj.weight"][:, rank * 2 : (rank + 1) * 2].clone(),
                    embed_name: expected[embed_name].clone(),
                }
                torch.save(
                    {
                        "module": module,
                        "mp_world_size": 2,
                        "optimizer_step": 60000,
                        UNIVERSAL_CHECKPOINT_INFO: info,
                    },
                    checkpoint / f"mp_rank_{rank:02d}_model_states.pt",
                )
            result = export_checkpoint(
                checkpoint,
                template,
                output,
                tp_size=2,
                expected_parameters=sum(x.numel() for x in expected.values()),
            )
            self.assertEqual(
                result["layouts"],
                {"replicated": 1, "identical_full": 1, "concat_dim_0": 1, "concat_dim_1": 1},
            )
            with safe_open(
                output / "model-00001-of-00001.safetensors", framework="pt", device="cpu"
            ) as handle:
                for name, tensor in expected.items():
                    self.assertTrue(torch.equal(handle.get_tensor(name), tensor), name)

    def test_rejects_divergent_declared_replica(self) -> None:
        from training.export_true_tp_checkpoint_to_hf_cpu import merge_tensor
        with self.assertRaisesRegex(RuntimeError, "declared replicated"):
            merge_tensor(
                "norm.weight",
                [torch.ones(2), torch.zeros(2)],
                [__import__("re").compile(r"norm\.weight")],
                [],
            )

    def test_rejects_unexpected_identical_tp_tensor(self) -> None:
        from training.export_true_tp_checkpoint_to_hf_cpu import merge_tensor
        with self.assertRaisesRegex(RuntimeError, "unexpected identical"):
            merge_tensor("q_proj.weight", [torch.ones(2, 2), torch.ones(2, 2)], [], [])


if __name__ == "__main__":
    unittest.main()
