# Stage 0 Design Freeze

Status: complete as of `2026-04-20`; this document is the frozen boundary reference for the current YOCO baseline implementation.

This document freezes the implementation boundary for the first `YOCO-Llama` baseline in this repository.

## Baseline Identity

- project baseline name: `YOCO-Llama-8B-Init`
- initialization source: `meta-llama/Llama-3.1-8B-Instruct`
- architecture family: YOCO-style decoder-decoder
- implementation goal: pure YOCO-Llama baseline, not a Comb/YOCO hybrid

## Architecture Split

- total decoder layers: 32
- self-decoder layers: 16
- cross-decoder layers: 16
- self-decoder attention mechanism: sliding-window attention (`SWA`)
- self-decoder attention exclusion: do not use gated retention in the first baseline

Initial layer assignment:

- self-decoder uses source Llama layers `0..15`
- cross-decoder uses source Llama layers `16..31` as the closest initialization source for compatible submodules

## Self-Decoder Attention Choice

The first YOCO-Llama baseline should implement the self-decoder with
YOCO-style sliding-window attention (`SWA`), not gated retention.

Why this choice is frozen for baseline v1:

- the baseline is explicitly Llama-initialized
- `SWA` remains much closer to Llama's attention structure than gated retention
- `SWA` gives a cleaner path for weight transfer, RoPE reuse, GQA reuse, and
  later FlashAttention-style implementation work
- using gated retention in the first baseline would enlarge the implementation
  delta and make it harder to interpret baseline behavior

## Training Strategy

- parameter freezing: none
- optimization strategy: full-parameter training
- first implementation target: correctness before throughput optimization

Rationale:

- publicly visible YOCO materials describe pretraining from scratch and do not expose a partial-freeze recipe
- the closest matching strategy for this repository's baseline is to initialize from Llama and train all parameters

## External Model Interface

The first YOCO baseline should look like a standard causal LM from the outside.

Expected input contract:

- `input_ids`
- `position_ids`
- `shift_labels`
- `cu_seqlens_q`
- `max_seqlen_q`
- `past_key_values`
- `use_cache`

Expected first-version output contract:

- logits
- optional loss
- optional cache
- optional hidden states

## Explicitly Excluded Interfaces

The following Comb-specific interfaces are excluded from the first YOCO baseline:

- `chunk_ids`
- `chunk_model`
- `cross_attention_states` supplied by a separate chunk encoder
- `position_ids_k`
- `cu_seqlens_k`
- `cu_seqlens_chunk`
- `max_seqlen_k`
- `max_seqlen_chunk`

## Data Path Constraint

The training path must use ordinary decoder tokens only.

Allowed first-version sample fields:

- `input_ids`
- `shift_labels`
- `token_length`

Allowed first-version batch metadata:

- `position_ids`
- `cu_seqlens_q`
- `max_seqlen_q`

The YOCO baseline must not depend on chunk packing or chunk-history preprocessing.

## Cache Ownership Constraint

The first implementation must encode the following semantic rule:

- reusable history memory is owned by the self-decoder path
- the cross-decoder consumes reusable memory
- the cross-decoder must not reintroduce a redundant full second history cache path equivalent to ordinary decoder self-attention caching

This rule is the main architectural constraint that differentiates the YOCO baseline from simply stacking more decoder layers.

## File Layout Constraint

The YOCO baseline must be implemented independently from CombLlama.

Planned file family:

- `baselines/YOCO/models/YOCO.py`
- `baselines/YOCO/models/YOCO_megatron.py`
- `baselines/YOCO/scripts/init_yoco_from_llama.py`
- `baselines/YOCO/training/train_yoco_megatron.py`

Comb files should remain untouched unless a later stage requires a narrowly scoped shared utility extraction.

## Acceptance Criteria for Stage 0

Stage 0 is complete only if all of the following are true:

- the baseline can be described without referencing Comb's chunk encoder as a required component
- the layer split is frozen to `16 + 16`
- the self-decoder attention mechanism is frozen to `SWA`
- the training strategy is frozen to full-parameter training
- the external interface is frozen to a standard `input_ids`-based causal LM path
- excluded Comb-specific interfaces are explicitly listed in the repo
