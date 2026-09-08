from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from benchmarks.audit_cache_size_reproduction import cache_record


class CacheSizeReproductionTest(unittest.TestCase):
    def test_llama_figure_values(self) -> None:
        base = SimpleNamespace(
            dtype=torch.bfloat16,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=32,
        )
        comb = SimpleNamespace(cross_attention_layers=list(range(3, 32, 4)))
        result = cache_record(
            "meta-llama/Llama-3.1-8B-Instruct", base, comb, 1024
        )
        self.assertEqual(result["baseline_cache_mb"], 128.0)
        self.assertEqual(result["comb_cache_mb"], 32.0)
        self.assertEqual(result["retained_fraction"], 0.25)
        self.assertEqual(result["compression_factor"], 4.0)


if __name__ == "__main__":
    unittest.main()
