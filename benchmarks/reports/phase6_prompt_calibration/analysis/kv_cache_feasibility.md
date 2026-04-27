# CombLlama KV Cache Feasibility Note

Current benchmark inference caches CombLlama chunk encoder outputs once per prompt and reuses those cross-attention states during decoding.

What is already implemented:

1. Historical prompt tokens are packed into chunks once.
2. `model.chunk_model(...)` is run once during prefill.
3. The resulting cross-attention K/V states are reused for every generated token.
4. This avoids repeatedly encoding 10K-100K historical chunk tokens.

What is still missing:

1. `CombLlamaTextModel.forward` creates a `DynamicCache` when `use_cache=True`, but the custom self-attention path `_self_attn_forward` computes Q/K/V from the current input and calls `flash_attn_varlen_func` directly.
2. `_self_attn_forward` does not update `past_key_values` for self-attention layers.
3. `_self_attn_forward` also does not read previous self-attention keys/values during decode.
4. Cross-attention cache support exists for compressed chunk states, but that alone is insufficient for autoregressive token-by-token decoding because recent decoder self-attention still needs cached self K/V.

Conclusion:

The benchmark can now run `max_new_tokens=32` dev sweeps using cached chunk states. Do not switch to `use_cache=True` for self-attention until self-attention cache update and decode semantics are implemented and verified against no-cache generation.
