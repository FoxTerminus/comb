# CombLlama KV Cache Feasibility Note

Current benchmark inference uses a conservative decode path that re-runs the packed context for each generated token.

Reason:

1. `CombLlamaTextModel.forward` creates a `DynamicCache` when `use_cache=True`, but the custom self-attention path `_self_attn_forward` computes Q/K/V from the current input and calls `flash_attn_varlen_func` directly.
2. `_self_attn_forward` does not update `past_key_values` for self-attention layers.
3. `_self_attn_forward` also does not read previous self-attention keys/values during decode.
4. Cross-attention cache support exists for compressed chunk states, but that alone is insufficient for autoregressive token-by-token decoding because recent decoder self-attention still needs cached self K/V.

Conclusion:

Do not switch benchmark generation to `use_cache=True` until self-attention cache update and decode semantics are implemented and verified against no-cache generation.
