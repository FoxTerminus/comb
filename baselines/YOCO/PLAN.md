# YOCO-Llama Baseline Plan

## Progress Snapshot

Last synced: `2026-04-20`

Current status based on `./baselines/YOCO/PLAN.md`, `./baselines/YOCO/STAGE0_DESIGN.md`, and the code already present under `./baselines/YOCO`:

- Stage 0 is complete: the baseline boundary is frozen in `STAGE0_DESIGN.md`.
- Stage 1 is complete at the implementation level: the standalone YOCO model skeleton exists and can be instantiated from config.
- Stage 2 is complete: a GPU smoke run verified the `16 + 16` forward path, output logits shape, and finite loss.
- Stage 3 is complete: `use_cache=True` prefill + decode matches full forward within tolerance, and the dedicated YOCO cache path is verified.
- Stage 4 is complete: the initialization CLI, checkpoint reload, post-init forward, and loading-info reporting are all verified.
- Stage 5 is complete: `collate_fn_yoco` packs ordinary decoder tokens and packed batches run directly through the YOCO model.
- Stage 6 is complete: the training script now has a synthetic-data acceptance mode, a short run decreases loss, tiny-slice overfitting is observed, and checkpoint resume works.
- Stage 7 is complete: TP adaptation runs on 2 GPUs, rank losses match, TP cache works, and TP output aligns with the non-TP reference within expected `bf16` tolerance.
- Stage 8 is complete: `prepare_inputs_for_generation` and `_reorder_cache` are implemented, greedy generation works with `use_cache=True`, and manual stepwise decoding matches `generate()`.
- Stage 9 is complete for the current correctness-stage baseline: a first verification report now exists in `./baselines/YOCO/VERIFICATION_REPORT.md` with smoke, cache, overfit, memory, latency, and Llama/CombLlama comparison notes.

## Objective

Build a pure YOCO-Llama baseline on top of `Llama-3.1-8B-Instruct` inside this repository.

This baseline should:

- implement a YOCO-style `self-decoder + cross-decoder` architecture
- use `Llama-3.1-8B-Instruct` as the initialization source
- use a `16 + 16` layer split
- follow the publicly visible YOCO training strategy as closely as possible in terms of parameter freezing, i.e. full-parameter training
- avoid mixing in the current `CombLlama` chunk encoder design
- first prioritize architectural correctness, cache correctness, training viability, and inference viability

This baseline should not, in the first stage:

- reuse the `chunk_ids -> chunk_model -> cross_attention_states` path
- reproduce the full official YOCO training recipe end-to-end
- target exact paper metrics before the architecture and cache behavior are verified

## Naming

Use a neutral internal name until parameter count is verified:

- base model: `Llama-3.1-8B-Instruct`
- experimental model name: `YOCO-Llama-8B-Init`

## Confirmed Design Decisions

- baseline type: pure YOCO-Llama baseline
- initialization source: `Llama-3.1-8B-Instruct`
- layer split: `16 self-decoder + 16 cross-decoder`
- self-decoder attention: YOCO-style `SWA` for the first baseline implementation
- self-decoder attention exclusions: do not use `gated retention` in the first baseline implementation
- freezing strategy: no freezing, full-parameter training
- external training interface: standard causal-LM-style `input_ids` path
- current Comb chunk path: excluded from this baseline

## Self-Decoder Attention Choice

For this repository's first YOCO-Llama baseline, the self-decoder should use
the paper's sliding-window attention (`SWA`) path rather than gated retention.

Reasoning:

- this baseline is defined as a Llama-initialized YOCO implementation
- `SWA` stays much closer to Llama attention structure, RoPE usage, GQA layout,
  and weight-transfer logic
- `gated retention` would require a larger architectural departure from Llama,
  a less natural parameter mapping, and a more invasive cache/kernel rewrite
- `SWA` is therefore the safer and clearer baseline-v1 choice

## Why Full-Parameter Training

Based on the publicly available YOCO materials, the exposed training path is pretraining-from-scratch and does not describe a partial-freeze strategy. The closest matching strategy for this repository's YOCO-Llama baseline is therefore:

- initialize from Llama weights
- train all parameters

Reference links:

- Microsoft Research paper page: <https://www.microsoft.com/en-us/research/publication/you-only-cache-once-decoder-decoder-architectures-for-language-models/>
- YOCO README: <https://github.com/microsoft/unilm/tree/master/YOCO>
- YOCO pretraining section: <https://github.com/microsoft/unilm/tree/master/YOCO#pretraining-from-scratch>

## Scope Boundary

### In Scope

- a new YOCO model family implemented independently from `CombLlama`
- YOCO-style two-stage decoder design
- YOCO-style cache ownership and reuse
- Llama-to-YOCO weight initialization
- training path using ordinary `input_ids`
- inference path with cache-enabled generation
- later TP adaptation

### Out of Scope for First Version

- chunk encoder variants
- historical chunk compression
- exact official YOCO dataset/mixture reproduction
- exact official YOCO training schedule reproduction
- immediate optimization for best throughput before correctness is verified

## Milestone Plan

### Stage 0: Freeze the Design

#### Goal

Lock the baseline boundary so implementation does not drift into a Comb/YOCO hybrid.

#### TODO

- [x] Fix the model identity as `YOCO-Llama-8B-Init`
- [x] Fix the layer split to `16 self + 16 cross`
- [x] Fix the training strategy to full-parameter training
- [x] Fix the external interface to standard `input_ids`-based causal LM inputs
- [x] Explicitly exclude `chunk_ids`, `chunk_model`, and chunk-specific packing from this baseline

#### Acceptance

- [x] A design note exists and clearly states that this baseline is a pure YOCO-Llama model
- [x] The model can be described in one sentence without referencing Comb's chunk encoder

### Stage 1: Create the YOCO Model Skeleton

#### Goal

Create an isolated YOCO implementation inside `./baselines/YOCO` without
modifying `models/CombLlama.py`.

#### TODO

- [x] Add `baselines/YOCO/models/YOCO.py`
- [x] Add `YOCOConfig`
- [x] Add `YOCOPreTrainedModel`
- [x] Add `YOCOTextModel`
- [x] Add `YOCOForCausalLM`
- [x] Keep HuggingFace-style outputs using `BaseModelOutputWithPast` and `CausalLMOutputWithPast`

#### Suggested Class Layout

- `YOCOConfig`
- `YOCOSelfDecoder`
- `YOCOCrossDecoder`
- `YOCOTextModel`
- `YOCOForCausalLM`
- `YOCODynamicCache` or an equivalent dedicated cache wrapper

#### Acceptance

- [x] The model can be instantiated from config
- [x] The printed module tree clearly separates self-decoder and cross-decoder

### Stage 2: Implement the 16+16 Forward Structure

#### Goal

Make the architecture run correctly before worrying about speed or distributed training.

#### Design

- first 16 layers act as the self-decoder
- last 16 layers act as the cross-decoder
- self-decoder attention should be implemented as YOCO-style `SWA`
- embeddings, RMSNorm, rotary embeddings, and MLP style should stay close to Llama where possible
- the cross-decoder should consume memory from the self-decoder rather than reuse the Comb chunk interface

#### TODO

- [x] Implement self-decoder forward
- [x] Implement cross-decoder layers
- [x] Implement `YOCOTextModel.forward(...)`
- [x] Implement `YOCOForCausalLM.forward(...)`
- [x] Compute logits and LM loss correctly

#### Acceptance

- [x] Random input forward passes without error
- [x] Output logits have correct shape
- [x] Loss is finite
- [x] No chunk-related input is required

### Stage 3: Implement YOCO Cache Semantics

#### Goal

Implement the main YOCO idea: reuse memory/cache correctly instead of maintaining a full second history cache path.

#### TODO

- [x] Define a dedicated YOCO cache structure
- [x] Implement prefill behavior
- [x] Implement decode behavior
- [x] Ensure self-decoder owns the reusable history memory
- [x] Ensure cross-decoder reads memory rather than maintaining a redundant full history self-attention cache
- [x] Support `use_cache=True`

#### Acceptance

- [x] `use_cache=False` works
- [x] `use_cache=True` works
- [x] Prefill followed by token-by-token decode works
- [x] Token-by-token decode approximately matches full forward logits within expected tolerance

### Stage 4: Llama-to-YOCO Weight Initialization

#### Goal

Initialize the YOCO baseline from `Llama-3.1-8B-Instruct` rather than random initialization.

#### Mapping Plan

- embeddings: copy from Llama
- self-decoder 16 layers: copy from Llama layers `0..15`
- cross-decoder 16 layers:
  - reuse Llama layers `16..31` for compatible norms and MLP blocks
  - initialize new cross-attention projections from corresponding self-attention projections where structurally sensible
  - initialize any newly introduced gating or residual scaling conservatively

#### TODO

- [x] Add `baselines/YOCO/scripts/init_yoco_from_llama.py`
- [x] Load `Llama-3.1-8B-Instruct`
- [x] Build the YOCO model
- [x] Apply a deterministic weight mapping
- [x] Save a new HuggingFace-style checkpoint
- [x] Print missing keys, unexpected keys, and total parameter count

#### Acceptance

- [x] The initialization script completes successfully
- [x] The saved checkpoint can be reloaded
- [x] Forward pass works after initialization
- [x] Parameter count is recorded

### Stage 5: Build the YOCO Data Path

#### Goal

Train the YOCO baseline with ordinary causal-LM-style inputs rather than Comb's chunk route.

#### Input Contract

The first training version should consume only the fields required by a standard decoder model:

- `input_ids`
- `shift_labels`
- `position_ids`
- `cu_seqlens_q`
- `max_seqlen_q`

#### TODO

- [x] Add `collate_fn_yoco`
- [x] Reuse current datasets only for their `input_ids` and `shift_labels`
- [x] Ignore `chunk_ids` during YOCO training
- [x] Ensure the collate path remains compatible with packed variable-length training if needed

#### Acceptance

- [x] A DataLoader batch can be passed directly into the YOCO model
- [x] No part of YOCO training requires `chunk_ids`

### Stage 6: Add the Training Script

#### Goal

Train the YOCO baseline within this repository using the current training stack as a template.

#### TODO

- [x] Add `baselines/YOCO/training/train_yoco_megatron.py`
- [x] Build the YOCO model from config or Llama initialization
- [x] Use full-parameter training
- [x] Reuse optimizer, scheduler, bf16, and gradient accumulation patterns from the current training script where appropriate
- [x] Add checkpoint save/load logic

#### Important Constraint

Do not freeze any parameter groups in the YOCO baseline training script.

#### Acceptance

- [x] Single-device training runs for at least one batch
- [x] Multi-step training produces decreasing loss on a small sample
- [x] Small-scale overfitting on a tiny dataset slice is possible

### Stage 7: Add Tensor Parallel Support

#### Goal

Adapt the current TP infrastructure to the new YOCO model only after single-device correctness is established.

#### TODO

- [x] Add `baselines/YOCO/models/YOCO_megatron.py`
- [x] Shard self-decoder attention/MLP projections
- [x] Shard cross-decoder attention/MLP projections
- [x] Patch local head counts correctly
- [x] Verify cache behavior under TP

#### Acceptance

- [x] TP=1 matches the non-TP path
- [x] TP>1 runs without shape or synchronization issues
- [x] Loss values across ranks are consistent

### Stage 8: Add Inference and Generation

#### Goal

Make the baseline usable for cache-enabled generation, not just training.

#### TODO

- [x] Implement `prepare_inputs_for_generation`
- [x] Implement cache-aware decode inputs
- [x] Implement `reorder_cache` if needed
- [x] Support greedy generation with `use_cache=True`

#### Acceptance

- [x] Prompt-to-text generation works
- [x] Cache-enabled generation does not fail on long prompts
- [x] Manual stepwise decoding and `generate()` are behaviorally consistent

### Stage 9: Baseline Verification and Comparison

#### Goal

Verify that the baseline is correct enough to compare against Llama and CombLlama.

#### TODO

- [x] Record total parameter count
- [x] Record trainable parameter count
- [x] Run smoke tests
- [x] Run cache consistency tests
- [x] Run short overfit tests
- [x] Measure memory usage
- [x] Measure prefill latency
- [x] Measure decode latency
- [x] Compare against original Llama and current CombLlama

#### Acceptance

- [x] A first comparison report exists
- [x] The baseline can be described in terms of correctness, training viability, and performance tradeoffs

## File-Level TODO List

### New Files

- [x] `baselines/YOCO/models/YOCO.py`
- [x] `baselines/YOCO/models/YOCO_megatron.py`
- [x] `baselines/YOCO/scripts/init_yoco_from_llama.py`
- [x] `baselines/YOCO/training/train_yoco_megatron.py`
- [x] `baselines/YOCO/tests/test_yoco_smoke.py`
- [x] `baselines/YOCO/tests/test_yoco_cache.py`
- [x] `baselines/YOCO/tests/test_yoco_init_from_llama.py`

### Data / Collate Changes

- [x] Add `collate_fn_yoco` in `data/base.py` or a dedicated YOCO collate file
- [x] Keep the YOCO path independent from chunk-specific batch fields

## Recommended Execution Order

1. implement `baselines/YOCO/models/YOCO.py`
2. run non-cache forward successfully
3. implement cache semantics
4. add `baselines/YOCO/scripts/init_yoco_from_llama.py`
5. add smoke tests
6. add `collate_fn_yoco`
7. add `baselines/YOCO/training/train_yoco_megatron.py`
8. verify small-sample overfitting
9. add `baselines/YOCO/models/YOCO_megatron.py`
10. enable TP training
11. add generation support
12. run comparison benchmarks

## Milestone Gates

### M1: Structure Complete

- [x] The model instantiates
- [x] Forward runs
- [x] Logits and loss are valid

### M2: Cache Complete

- [x] Prefill and decode both work
- [x] Cache semantics are stable
- [x] Stepwise decode and full forward align within tolerance

### M3: Initialization Complete

- [x] Llama-to-YOCO initialization works
- [x] Reloaded checkpoints run correctly

### M4: Training Complete

- [x] Loss decreases on a small training run
- [x] Tiny-slice overfitting is possible

### M5: Parallel Complete

- [x] TP training works
- [x] Rank consistency checks pass

### M6: Baseline Complete

- [x] Generation works
- [x] Benchmark numbers exist
- [x] Comparison against Llama and CombLlama exists

## Current Default Implementation Choices

Unless later revised, the default first implementation should use:

- model name: `YOCO-Llama-8B-Init`
- split: `16 self + 16 cross`
- training style: full-parameter training
- input style: ordinary `input_ids + shift_labels`
- initialization: Llama-based weight transfer
- priority: single-device correctness first, TP later
