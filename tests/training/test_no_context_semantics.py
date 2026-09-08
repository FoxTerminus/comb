"""Regression tests for the no-context arm of PIC evaluation."""

from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig

from comb.integration.hf.CombLlama import (
    CombLlamaConfig,
    CombLlamaForConditionalGeneration,
)


def make_tiny_model() -> CombLlamaForConditionalGeneration:
    text_config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    config = CombLlamaConfig(
        text_config=text_config,
        chunk_token_index=127,
        num_hidden_layers=3,
        cross_attention_layers=[1],
        pad_token_id=0,
    )
    return CombLlamaForConditionalGeneration(config=config).eval()


class NoContextSemanticsTests(unittest.TestCase):
    def test_no_context_skips_cross_attention_and_has_no_stale_state(self):
        torch.manual_seed(42)
        model = make_tiny_model()
        input_ids = torch.randint(3, 120, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        chunk_ids = torch.randint(3, 120, (2, 11))
        cross_attention_mask = torch.ones_like(chunk_ids)

        cross_calls = 0

        def count_cross_call(_module, _inputs, _output):
            nonlocal cross_calls
            cross_calls += 1

        hook = model.language_model.model.cross_layers[0].register_forward_hook(
            count_cross_call
        )
        try:
            with torch.inference_mode():
                no_context_before = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                self.assertEqual(cross_calls, 0)

                model(
                    input_ids=input_ids,
                    chunk_ids=chunk_ids,
                    attention_mask=attention_mask,
                    cross_attention_mask=cross_attention_mask,
                    use_cache=False,
                )
                self.assertEqual(cross_calls, 1)

                no_context_after = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                self.assertEqual(cross_calls, 1)
        finally:
            hook.remove()

        self.assertTrue(torch.equal(no_context_before, no_context_after))


if __name__ == "__main__":
    unittest.main()
