# YOCO Verification Report

Last updated: `2026-04-20`

## Scope

This report records the first runnable verification pass for the current
`YOCO-Llama-8B-Init` baseline implementation.

All measurements below were collected on this repository using tiny synthetic
32-layer configurations for architectural verification. They are useful for
correctness and relative behavior checks, not for final production claims.

## What Was Verified

### Stage 2: Forward Path

- `YOCOForCausalLM` runs a full `16 + 16` forward pass on GPU.
- Logits shape is correct.
- LM loss is finite.

Observed smoke result:

- logits shape: `(1, 8, 128)`
- loss: `4.890026`

### Stage 3: Cache Semantics

- `use_cache=False` runs successfully.
- `use_cache=True` runs successfully.
- Prefill followed by token-by-token decode runs successfully.
- Stepwise decode matches full-forward last-token logits within tolerance.

Observed cache result:

- full-cache sequence length: `8`
- prefill-cache sequence length before decode: `8`
- decode-cache sequence length after decode: `8`
- max absolute difference between full forward and stepwise decode: `0.0`

### Stage 4: Llama-to-YOCO Initialization

- The real CLI entrypoint `baselines/YOCO/scripts/init_yoco_from_llama.py` completes successfully against a tiny local Llama checkpoint.
- The saved checkpoint reloads correctly.
- Post-init forward works.
- The script now records and prints:
  - `total_params`
  - `trainable_params`
  - `missing_keys`
  - `unexpected_keys`

Observed initialization result:

- total parameters: `2,117,696`
- trainable parameters: `2,117,696`
- missing keys: `[]`
- unexpected keys: `[]`

### Stage 5: Data Path

- `collate_fn_yoco` packs ordinary decoder tokens without requiring any chunk-specific fields.
- A packed `DataLoader` batch passes directly into the YOCO model.

Observed data-path result:

- packed input shape: `(1, 6)`
- packed logits shape: `(1, 6, 128)`
- packed loss: `4.895580`

### Stage 6: Training

- The official training script now supports a deterministic synthetic-data mode for smoke and overfit verification.
- Single-device training runs for at least one batch.
- The short synthetic run shows clear loss decrease.
- Checkpoint save/load and resume both work.

Observed synthetic overfit run on GPU:

- step 1 loss: `4.871312`
- step 12 loss: `2.672504`
- resumed step 13 loss: `2.638268`

Interpretation:

- the training path is viable
- the model can overfit a tiny repeated sample
- checkpoint resume preserves optimizer/scheduler/model state well enough to continue improving loss

### Stage 7: Tensor Parallel

- A 2-rank TP smoke run on `GPU 2` and `GPU 3` succeeds.
- Rank losses are consistent.
- TP cache path works.
- TP output matches the non-TP reference within expected `bf16` tolerance.

Observed TP result:

- TP world size: `2`
- reference vs TP max absolute difference: `0.01025390625`
- TP cache max absolute difference: `0.0`
- per-rank losses: `[4.755093, 4.755093]`

### Stage 8: Generation

- `prepare_inputs_for_generation` is implemented.
- `_reorder_cache` is implemented for the current single-sequence generation mode.
- Greedy generation with `use_cache=True` works.
- A longer prompt generation run succeeds.
- Manual stepwise decode and `generate()` produce identical tokens on the same prompt.

Observed generation result:

- prompt length: `16`
- generated total length: `19`
- `generate()` output exactly matched manual stepwise decoding

## Tiny Comparison Snapshot

Configuration:

- decoder layers: `32`
- hidden size: `64`
- attention heads: `8`
- key/value heads: `8`
- prompt length: `16`
- cross-memory length for CombLlama: `16`
- dtype: `bfloat16`

Measured values:

| Model | Params | Prefill ms | Decode ms | Peak MiB |
|---|---:|---:|---:|---:|
| YOCO | 2,117,696 | 25.324 | 25.926 | 20.883 |
| Llama | 2,117,696 | 35.348 | 31.489 | 20.947 |
| CombLlama | 2,347,656 | 25.111 | 24.059 | 20.841 |

## Current Readout

Correctness:

- The YOCO baseline is now architecturally runnable.
- Forward, cache, init, generation, training, and TP all have direct execution evidence.

Training viability:

- The model trains with the repository training stack.
- A tiny synthetic slice overfits quickly, which is the expected first viability signal.

Performance tradeoffs:

- On the tiny verification setup, YOCO prefill and decode are both faster than the equally sized tiny Llama reference.
- Tiny CombLlama remains competitive on these synthetic measurements, but it also carries extra cross-attention machinery and a different interface contract.
- These numbers should be treated as a correctness-stage comparison only; they are not final throughput claims for the real 8B checkpoints.
