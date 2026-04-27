# SambaY-Llama Baseline Plan

## Progress Snapshot

Last synced: `2026-04-27`

Current status:

- `baselines/SambaY/SambaY.pdf` is present locally.
- No SambaY model, initialization script, training path, TP adapter, tests, or benchmark config exists yet.
- The official implementation reference is `microsoft/ArchScale`, not the older standalone `microsoft/Samba` repository.
- The baseline target is `Llama-3.1-8B-Instruct` as backbone, built as a structural baseline against `CombLlama`.

## Objective

Build a SambaY-style `decoder-hybrid-decoder` baseline initialized from `Llama-3.1-8B-Instruct`.

The baseline should:

- follow the SambaY paper and `microsoft/ArchScale` implementation as the architectural source of truth
- use a Samba-style self-decoder with Mamba token mixers, sliding-window attention, and a single full-attention layer that produces the shared YOCO KV cache
- use a cross-decoder that reuses the shared KV cache and replaces about half of the cross-attention layers with GMU/nGMU blocks
- keep the external training interface as ordinary causal-LM `input_ids` rather than Comb chunk inputs
- support cache-enabled generation and the same benchmark harness used for YOCO and CombLlama

This baseline should not:

- reuse `chunk_ids`, `chunk_model`, or `cross_attention_states` from `CombLlama`
- implement a naive `Samba + YOCO` baseline and call it SambaY
- depend on exact Phi-4-mini-Flash hyperparameters when they conflict with the Llama-3.1-8B backbone
- target official paper metrics before architecture, cache, and training correctness are verified

## Source-of-Truth Notes

From the SambaY paper:

- SambaY applies GMUs to the YOCO cross-decoder and replaces half of the cross-attention layers.
- The self-decoder is Samba-based and provides two memory branches: one shared KV cache from the final full-attention layer, and one SSM kernel output state from the final Mamba layer.
- The GMU computes `Y = (M * SiLU(X W1^T)) W2`; nGMU adds RMSNorm after the element-wise multiplication.
- SambaY preserves linear prefill complexity because only one full-attention KV cache is materialized.
- Compared with YOCO, SambaY additionally caches the final SSM kernel output state `m`, with `d_mem = 2 * d_model` for Mamba-1.
- The paper reports that normalization placement is important for linear-attention/SSM memories; use nGMU for Mamba2/GDN-style variants, and keep the memory before normalization when applicable.

From `microsoft/ArchScale`:

- `Config` has `yoco=True`, `gmu_yoco=True`, `rnn_per_layer=2`, `rnn_type="mamba"`, `gmu_per_layer=2`, and `nope=True` for `sambay_d*`.
- YOCO mode splits total depth in half. The first half is the self-decoder. Around the boundary, the implementation creates a full-attention layer for shared KV, then later cross-decoder layers either use cross-attention or GMU.
- `CausalSelfAttention(..., yoco_cross=True)` owns only query and output projections; K/V come from the shared YOCO cache.
- `GMUWrapper` wraps `GMU(d_model, d_mem)` and passes `gmu_mems` through unchanged.
- Mamba saves `gmu_mems` from the SSM scan output before output projection when `gmu_save=True`; the ordinary Mamba output still goes through its gate and `out_proj`.

Reference links:

- Paper page: <https://arxiv.org/abs/2507.06607>
- Hugging Face paper page: <https://huggingface.co/papers/2507.06607>
- Official code: <https://github.com/microsoft/ArchScale>
- Original Samba codebase: <https://github.com/microsoft/Samba>

## Naming

- baseline name: `SambaY-Llama-8B-Init`
- module path: `baselines/SambaY`
- model type: `sambay_llama`
- comparison role: architecture baseline for `CombLlama`

## Architecture Decision

Use the official SambaY design, adapted conservatively to Llama-3.1-8B:

- total layers: 32, matching Llama-3.1-8B
- self-decoder: first 16 logical layers
- cross-decoder: last 16 logical layers
- self-decoder token mixers:
  - Mamba-1 on every second layer by default, matching `rnn_per_layer=2`
  - sliding-window attention on the remaining local layers
  - one full-attention layer at the self/cross boundary to generate the shared KV cache
- cross-decoder:
  - cross-attention layers reuse the single shared KV cache
  - GMU layers replace every other cross-attention layer, matching `gmu_per_layer=2`
- positional encoding:
  - default SambaY path uses NoPE for hybrid SSM architectures
  - if Llama weight initialization is too unstable under immediate NoPE conversion, keep a controlled ablation flag `use_rope_for_llama_init=True`, but the official baseline should report NoPE results separately
- memory dimensions:
  - `d_model = 4096` for Llama-3.1-8B
  - Mamba GMU memory `d_mem = 2 * d_model = 8192`
  - attention head dim remains 128 for Llama compatibility
  - GQA layout follows Llama for attention and cross-attention

## Weight Initialization Policy

Llama initialization is useful but only partially structural:

- embeddings, final norm, and LM head: copy directly from Llama
- Llama MLP/RMSNorm blocks: copy into corresponding SambaY layers where dimensions match
- SWA/full-attention layers: initialize from matching Llama self-attention projections
- cross-attention layers:
  - copy query/output projections from Llama attention
  - do not create per-layer K/V projections in yoco-cross layers
  - shared K/V projections come from the boundary full-attention layer
- Mamba layers: no direct Llama equivalent; initialize with official Mamba initialization, then train full-parameter
- GMU layers: initialize `in_proj` and `out_proj` with the repository default linear initialization; zero biases when present
- all parameters remain trainable

## Milestone Plan

### Stage 0: Freeze Baseline Boundary

Goal: prevent SambaY from drifting into CombLlama or naive Samba+YOCO.

TODO:

- [ ] Create `STAGE0_DESIGN.md`
- [ ] Freeze `SambaY-Llama-8B-Init` as a plain causal-LM baseline
- [ ] Explicitly exclude `chunk_ids`, `chunk_model`, and Comb cross-attention states
- [ ] Define the official variant as `Mamba-1 + SWA + one shared full-attn KV + interleaved GMU/cross-attn`
- [ ] Define ablations separately: `Samba+YOCO`, `SWA+YOCO`, `SambaY+RoPE`, `SambaY+DA`

Acceptance:

- [ ] One-page design note describes SambaY without referencing Comb internals
- [ ] Model can be summarized as `decoder-hybrid-decoder with shared KV and GMU memory`

### Stage 1: Implement Core Model Skeleton

Goal: add an isolated HuggingFace-style model family.

TODO:

- [ ] Add `baselines/SambaY/models/SambaY.py`
- [ ] Add `SambaYConfig`
- [ ] Add `SambaYPreTrainedModel`
- [ ] Add `SambaYSelfDecoder`
- [ ] Add `SambaYCrossDecoder`
- [ ] Add `SambaYTextModel`
- [ ] Add `SambaYForCausalLM`
- [ ] Add `SambaYDynamicCache`

Acceptance:

- [ ] Model instantiates from a tiny config
- [ ] Module tree clearly separates self-decoder, shared KV producer, GMU layers, and cross-attention layers

### Stage 2: Implement Samba Self-Decoder

Goal: reproduce the SambaY self-decoder branch that produces both KV and GMU memory.

TODO:

- [ ] Implement or vendor a minimal Mamba-1 layer compatible with current training dependencies
- [ ] Implement Mamba `gmu_save` behavior: save SSM scan output before Mamba output projection
- [ ] Implement sliding-window attention layers
- [ ] Implement the boundary full-attention layer that materializes shared K/V
- [ ] Keep attention dimensions compatible with Llama GQA
- [ ] Add NoPE mode as the official default

Acceptance:

- [ ] Self-decoder forward returns hidden states, shared KV cache, and `gmu_mems`
- [ ] Tiny forward pass has finite logits/loss
- [ ] Disabling GMU still leaves a valid Samba+YOCO ablation path

### Stage 3: Implement GMU/nGMU Cross-Decoder

Goal: implement the actual SambaY distinction from Samba+YOCO.

TODO:

- [ ] Add `GMU` and `GMUWrapper`
- [ ] Implement `GMU(hidden, memory) = out_proj(SwiGLU(in_proj(hidden), memory))`
- [ ] Add optional nGMU with RMSNorm after output gating for future Mamba2/GDN variants
- [ ] Interleave GMU and cross-attention layers in the cross-decoder
- [ ] Ensure GMU receives the self-decoder SSM memory, not Comb chunk states
- [ ] Ensure cross-attention layers own Q/O only and read K/V from shared cache

Acceptance:

- [ ] GMU layers run with `d_mem = 8192` for Llama-8B hidden size
- [ ] Cross-attention layers reuse exactly one shared KV cache
- [ ] No cross-decoder layer maintains a private full-history K/V cache

### Stage 4: Cache Semantics and Generation

Goal: support prefill plus token-by-token decode.

TODO:

- [ ] Define `SambaYDynamicCache(shared_kv, gmu_mems, mamba_states, conv_states, seen_tokens)`
- [ ] During prefill, compute self-decoder once and store shared KV plus GMU memory
- [ ] During decode, update recurrent Mamba states and reuse shared prompt KV where correct
- [ ] Implement `prepare_inputs_for_generation`
- [ ] Implement `_reorder_cache` for batch size 1 first; document limitations
- [ ] Add greedy generation smoke path

Acceptance:

- [ ] `use_cache=False` forward works
- [ ] `use_cache=True` prefill + decode works
- [ ] Manual stepwise decode approximately matches full forward on short prompts within expected bf16 tolerance
- [ ] Long prompt generation does not allocate per-layer cross-decoder KV histories

### Stage 5: Llama-to-SambaY Initialization

Goal: create a reproducible initialization script.

TODO:

- [ ] Add `baselines/SambaY/scripts/init_sambay_from_llama.py`
- [ ] Load `meta-llama/Llama-3.1-8B-Instruct` or a local checkpoint path
- [ ] Build `SambaYConfig` from Llama config
- [ ] Apply deterministic partial mapping
- [ ] Print loaded, missing, unexpected, and newly initialized parameter groups
- [ ] Save a HuggingFace-style checkpoint

Acceptance:

- [ ] Script completes on a tiny local Llama fixture
- [ ] Saved checkpoint reloads
- [ ] Forward pass after initialization produces finite loss
- [ ] Parameter and trainable-parameter counts are recorded

### Stage 6: Data and Training Path

Goal: train SambaY as a clean causal-LM baseline.

Important boundary:

- SambaY may use the same raw datasets as CombLlama and YOCO.
- SambaY must still own a dedicated preprocessing/collate function because its input contract is different from Comb's chunk path and may diverge from YOCO once Mamba state, NoPE, variable-length packing, and cache-prefill metadata are handled.
- Do not pass through unused Comb fields just because the shared dataset objects expose them.

TODO:

- [ ] Add `baselines/SambaY/training/data.py`
- [ ] Implement `preprocess_sambay_example(...)` for raw examples from the shared datasets
- [ ] Implement `collate_fn_sambay(...)` for packed SambaY batches
- [ ] Keep dataset selection shared, but keep token packing and batch construction SambaY-specific
- [ ] Consume `input_ids`, `shift_labels`, `attention_mask` or packed sequence metadata, `cu_seqlens_q`, `max_seqlen_q`
- [ ] Make `position_ids` optional because official SambaY uses NoPE; include it only for the controlled RoPE ablation path
- [ ] Produce reset/sequence-boundary metadata needed by Mamba and variable-length packed training
- [ ] Ensure labels are ordinary next-token LM labels and masking matches the chat/SFT format
- [ ] Ignore all Comb chunk-specific fields
- [ ] Add `baselines/SambaY/training/train_sambay_megatron.py`
- [ ] Use full-parameter training
- [ ] Reuse current optimizer/scheduler/bf16/checkpoint patterns

Acceptance:

- [ ] Shared raw datasets can be loaded through the SambaY preprocessing path
- [ ] A preprocessed SambaY example contains only fields required by SambaY
- [ ] A `collate_fn_sambay` batch runs through the model without `chunk_ids`
- [ ] Variable-length packed batches expose correct sequence-boundary metadata for Mamba/SWA attention
- [ ] Single-device synthetic batch trains
- [ ] Loss decreases on a tiny sample
- [ ] Tiny-slice overfit is possible
- [ ] Checkpoint save/resume works

### Stage 7: Tensor Parallel Support

Goal: make SambaY comparable to YOCO and CombLlama under the current multi-GPU setup.

TODO:

- [ ] Add `baselines/SambaY/models/SambaY_megatron.py`
- [ ] Shard attention Q/O and shared K/V projections
- [ ] Shard MLP projections
- [ ] Shard GMU `in_proj`/`out_proj`
- [ ] Handle Mamba projections and convolution state carefully
- [ ] Verify TP cache behavior

Acceptance:

- [ ] TP=1 matches non-TP
- [ ] TP=2 runs forward/backward
- [ ] Rank losses match within expected tolerance
- [ ] Cache-enabled generation runs under TP for batch size 1

### Stage 8: Tests and Verification

Goal: make failures local and easy to diagnose.

TODO:

- [ ] Add `tests/test_sambay_smoke.py`
- [ ] Add `tests/test_sambay_gmu.py`
- [ ] Add `tests/test_sambay_cache.py`
- [ ] Add `tests/test_sambay_init_from_llama.py`
- [ ] Add `tests/test_sambay_training.py`
- [ ] Add `VERIFICATION_REPORT.md`

Acceptance:

- [ ] Forward/loss test passes
- [ ] GMU shape and gradient test passes
- [ ] Cache consistency test passes
- [ ] Initialization reload test passes
- [ ] Training smoke and overfit tests pass

### Stage 9: Benchmark Integration

Goal: compare SambaY against Llama, YOCO, and CombLlama on the same harness.

TODO:

- [ ] Add `benchmarks/configs/sambay_dev.json`
- [ ] Add `SambaY` to shared benchmark model loader
- [ ] Run RULER smoke at 4K/8K/16K/32K
- [ ] Run SCBench cache-memory and latency measurements
- [ ] Run LongBench, LongCodeBench, and LoCoMo only after cache/generation are stable
- [ ] Report quality and efficiency in the same schema as YOCO/CombLlama

Required result columns:

```text
model, checkpoint, benchmark, task, context_length, compression_ratio,
quality_metric, peak_memory_gb, prefill_latency_s, decode_latency_s,
tokens_per_second, run_time, git_commit, notes
```

Acceptance:

- [ ] SambaY dev predictions exist
- [ ] Peak memory, prefill latency, decode latency, and tokens/s are recorded
- [ ] Comparison table includes `Llama`, `YOCO`, `SambaY`, and `CombLlama`

## File-Level TODO List

New files:

- [ ] `baselines/SambaY/STAGE0_DESIGN.md`
- [ ] `baselines/SambaY/README.md`
- [ ] `baselines/SambaY/models/SambaY.py`
- [ ] `baselines/SambaY/models/SambaY_megatron.py`
- [ ] `baselines/SambaY/scripts/init_sambay_from_llama.py`
- [ ] `baselines/SambaY/training/data.py`
- [ ] `baselines/SambaY/tests/test_sambay_data.py`
- [ ] `baselines/SambaY/training/train_sambay_megatron.py`
- [ ] `baselines/SambaY/tests/test_sambay_smoke.py`
- [ ] `baselines/SambaY/tests/test_sambay_gmu.py`
- [ ] `baselines/SambaY/tests/test_sambay_cache.py`
- [ ] `baselines/SambaY/tests/test_sambay_init_from_llama.py`
- [ ] `baselines/SambaY/VERIFICATION_REPORT.md`
- [ ] `benchmarks/configs/sambay_dev.json`

Likely shared changes:

- [ ] `baselines/SambaY/__init__.py`
- [ ] `baselines/SambaY/models/__init__.py`
- [ ] benchmark model loading utilities
- [ ] optional benchmark report aggregation

## Recommended Execution Order

1. Write `STAGE0_DESIGN.md`.
2. Implement tiny-config `SambaYConfig` and model skeleton.
3. Implement GMU first and unit-test it in isolation.
4. Implement self-decoder with Mamba and shared KV output.
5. Implement cross-decoder interleaving GMU and cross-attention.
6. Verify non-cache forward and loss.
7. Implement `SambaYDynamicCache`.
8. Verify prefill + decode consistency.
9. Add Llama-to-SambaY initialization.
10. Add training smoke and tiny overfit.
11. Add TP only after single-device correctness.
12. Integrate benchmark config and run dev comparisons.

## Milestone Gates

### M1: Architecture Complete

- [ ] model instantiates
- [ ] forward/loss works
- [ ] GMU and shared KV paths are both active

### M2: Cache Complete

- [ ] prefill and decode work
- [ ] cross-decoder uses one shared KV cache
- [ ] GMU memory is cached separately from KV

### M3: Initialization Complete

- [ ] Llama-to-SambaY script works
- [ ] checkpoint reload works
- [ ] missing/new parameters are documented

### M4: Training Complete

- [ ] loss decreases
- [ ] tiny-slice overfit works
- [ ] resume works

### M5: Parallel Complete

- [ ] TP forward/backward works
- [ ] TP cache works
- [ ] rank consistency checks pass

### M6: Baseline Complete

- [ ] generation works
- [ ] dev benchmark numbers exist
- [ ] comparison against YOCO and CombLlama is recorded

## Risks and Mitigations

- Mamba dependency risk: use the installed `mamba_ssm`/`causal_conv1d` path if available; otherwise add a slow correctness fallback for tests.
- Llama initialization mismatch: treat Mamba and GMU as newly initialized modules and require a short full-parameter adaptation stage before benchmarking.
- NoPE transition risk: keep a RoPE ablation flag for diagnosis, but do not let it replace the official SambaY baseline.
- Cache correctness risk: write cache tests before TP; debugging TP cache first will be unnecessarily painful.
- Benchmark fairness risk: report SambaY as a full-parameter trained architecture baseline, not as a compression-ratio variant unless explicit cache pruning is later added.
