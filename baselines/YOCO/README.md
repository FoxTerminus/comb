# YOCO Baseline

## Overview

This folder contains the repository's first pure YOCO-style baseline built on
top of a Llama-compatible backbone.

The goal of this baseline is not to reproduce the full official YOCO training
recipe immediately. The goal is to provide a clean, inspectable, runnable
baseline inside this repository that:

- stays independent from the existing `CombLlama` chunk-encoder path
- implements a YOCO-style `self-decoder + cross-decoder` split 
- initializes from `Llama-3.1-8B-Instruct`
- supports ordinary decoder-only training inputs
- supports cache-enabled generation
- is easy to compare against both plain Llama and `CombLlama`

The current design freeze is documented in
[STAGE0_DESIGN.md](/data3/junhaohu/comb/baselines/YOCO/STAGE0_DESIGN.md),
the implementation checklist is tracked in
[PLAN.md](/data3/junhaohu/comb/baselines/YOCO/PLAN.md),
and the first verification pass is recorded in
[VERIFICATION_REPORT.md](/data3/junhaohu/comb/baselines/YOCO/VERIFICATION_REPORT.md).

## What YOCO Means Here

In this repository, YOCO is implemented as a two-stage decoder:

- the first 16 layers form the self-decoder
- the last 16 layers form the cross-decoder
- the self-decoder computes local token interactions with sliding-window attention
- the self-decoder also owns reusable history memory
- the cross-decoder reads that memory instead of running a second full-history self-attention path

That is the main architectural distinction relative to a standard decoder-only
Llama stack. It is also the main reason this baseline exists: to test whether
the YOCO-style memory split is correct, trainable, and useful in this codebase.

## Design Choices

### Layer Split

The current baseline is fixed to:

- `16` self-decoder layers
- `16` cross-decoder layers

The total remains `32` layers to align naturally with
`Llama-3.1-8B-Instruct`.

### Self-Decoder Attention

The self-decoder uses YOCO-style sliding-window attention, not gated
retention.

This choice keeps the baseline close to Llama in:

- RoPE usage
- grouped-query attention shape
- linear projection layout
- weight transfer logic

### Training Strategy

The current baseline uses full-parameter training:

- initialize from Llama
- train all parameters
- do not freeze subsets of the model

### External Interface

The model is intentionally exposed like a standard causal LM.

The main first-version batch contract is:

- `input_ids`
- `shift_labels` or `labels`
- `position_ids`
- `cu_seqlens_q`
- `max_seqlen_q`

Optional generation-time fields:

- `past_key_values`
- `use_cache`

Excluded Comb-specific fields:

- `chunk_ids`
- `chunk_model`
- `cross_attention_states`
- `position_ids_k`
- `cu_seqlens_k`
- `cu_seqlens_chunk`
- `max_seqlen_k`
- `max_seqlen_chunk`

## Directory Layout

- [models/YOCO.py](/data3/junhaohu/comb/baselines/YOCO/models/YOCO.py): core YOCO model, cache, generation hooks
- [models/YOCO_megatron.py](/data3/junhaohu/comb/baselines/YOCO/models/YOCO_megatron.py): tensor-parallel adaptation
- [scripts/init_yoco_from_llama.py](/data3/junhaohu/comb/baselines/YOCO/scripts/init_yoco_from_llama.py): weight transfer from Llama to YOCO
- [training/data.py](/data3/junhaohu/comb/baselines/YOCO/training/data.py): YOCO-only collate path
- [training/train_yoco_megatron.py](/data3/junhaohu/comb/baselines/YOCO/training/train_yoco_megatron.py): distributed training entrypoint
- [tests/test_yoco_smoke.py](/data3/junhaohu/comb/baselines/YOCO/tests/test_yoco_smoke.py): forward and generation smoke tests
- [tests/test_yoco_cache.py](/data3/junhaohu/comb/baselines/YOCO/tests/test_yoco_cache.py): cache correctness tests
- [tests/test_yoco_init_from_llama.py](/data3/junhaohu/comb/baselines/YOCO/tests/test_yoco_init_from_llama.py): initialization tests
- [STAGE0_DESIGN.md](/data3/junhaohu/comb/baselines/YOCO/STAGE0_DESIGN.md): design freeze
- [PLAN.md](/data3/junhaohu/comb/baselines/YOCO/PLAN.md): implementation and milestone checklist
- [VERIFICATION_REPORT.md](/data3/junhaohu/comb/baselines/YOCO/VERIFICATION_REPORT.md): first runtime verification summary

## Model Structure

### `YOCOConfig`

`YOCOConfig` wraps a Llama text config and adds the split-specific fields:

- `num_self_decoder_layers`
- `num_cross_decoder_layers`

The sum must equal `text_config.num_hidden_layers`.

### `YOCOSlidingWindowAttention`

This module is the self-decoder attention operator.

Responsibilities:

- project `q`, `k`, `v`
- apply RoPE
- run FlashAttention in sliding-window causal mode during standard forward
- keep only the recent window in the self-decoder KV cache during generation

### `YOCODynamicCache`

This is the YOCO-specific cache object used by generation.

It stores:

- one sliding-window KV cache per self-decoder layer
- the accumulated self-decoder hidden states that act as cross-decoder memory
- the matching memory position ids

Current scope:

- single packed sequence per generation step
- generation-oriented cache path

### `YOCOSelfDecoder`

This stack contains the first half of layers.

Responsibilities:

- run sliding-window self-attention
- update the self-decoder cache
- produce the hidden states that will later be treated as reusable memory

### `YOCOCrossDecoder`

This stack contains the second half of layers.

Responsibilities:

- attend from current query states to self-decoder memory
- avoid maintaining a redundant full-history self-attention cache path
- produce the final decoder states before the LM head

### `YOCOTextModel`

This module ties everything together:

- token embeddings
- RoPE
- self-decoder
- cross-decoder
- final RMSNorm

Important semantic point:

- when `use_cache=False`, the full prompt is processed causally
- when `use_cache=True`, the first prefill call still preserves causal prompt semantics
- subsequent decode steps consume the accumulated self-decoder memory through the YOCO cache object

### `YOCOForCausalLM`

This is the public HF-style wrapper:

- forwards to `YOCOTextModel`
- applies the LM head
- computes causal LM loss
- implements `prepare_inputs_for_generation`
- supports greedy `generate()` with the custom YOCO cache

## Initialization From Llama

The baseline is initialized from a Llama checkpoint by
[init_yoco_from_llama.py](/data3/junhaohu/comb/baselines/YOCO/scripts/init_yoco_from_llama.py).

Current mapping:

- embeddings <- Llama embeddings
- final norm <- Llama final norm
- lm head <- Llama lm head
- self-decoder layers `0..15` <- Llama layers `0..15`
- cross-decoder norms and MLPs <- Llama layers `16..31`
- cross-decoder cross-attention projections <- Llama self-attention projections from layers `16..31`

The script also:

- saves a HF-style checkpoint
- reloads that checkpoint
- records `total_params`
- records `trainable_params`
- records `missing_keys`
- records `unexpected_keys`

### Example

```bash
python ./baselines/YOCO/scripts/init_yoco_from_llama.py \
  --llama-path meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /path/to/YOCO-Llama-8B-Init \
  --dtype bfloat16
```

## Training Data Path

YOCO training uses the collate function in
[training/data.py](/data3/junhaohu/comb/baselines/YOCO/training/data.py).

It packs variable-length decoder-only samples into a single continuous sequence
for FlashAttention-style execution.

Expected per-sample fields:

- `input_ids`
- `shift_labels`

Produced batch fields:

- `input_ids`
- `shift_labels`
- `position_ids`
- `cu_seqlens_q`
- `max_seqlen_q`

It intentionally ignores `chunk_ids` and all other Comb-specific fields.

## Training Script

The main training entrypoint is
[training/train_yoco_megatron.py](/data3/junhaohu/comb/baselines/YOCO/training/train_yoco_megatron.py).

Capabilities:

- single-GPU training
- DP + TP process-group setup
- optional initialization from a YOCO checkpoint
- checkpoint save and resume
- bf16 execution
- deterministic synthetic-data mode for smoke and overfit validation

### Common Arguments

- `--model-name`: Llama checkpoint path or HF model id
- `--init-yoco-path`: pre-initialized YOCO checkpoint
- `--tp-size`: tensor parallel size
- `--global-batch-size`
- `--micro-batch-size`
- `--lr`
- `--warmup-steps`
- `--total-steps`
- `--bf16` / `--no-bf16`
- `--resume-ckpt`
- `--output-dir`
- `--ckpt-dir`

### Synthetic Smoke / Overfit Run

Use this first. It verifies that the training path works before spending time
on real datasets.

```bash
PYTHONPATH=/data3/junhaohu/comb \
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  ./baselines/YOCO/training/train_yoco_megatron.py \
  --tp-size 1 \
  --model-name /path/to/tiny-or-real-llama \
  --synthetic-data \
  --synthetic-num-samples 64 \
  --synthetic-seq-len 16 \
  --global-batch-size 1 \
  --micro-batch-size 1 \
  --lr 0.02 \
  --warmup-steps 1 \
  --total-steps 64 \
  --max-steps-per-dataset 32
```

### Training From A YOCO Initialization

```bash
PYTHONPATH=/data3/junhaohu/comb \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  ./baselines/YOCO/training/train_yoco_megatron.py \
  --tp-size 2 \
  --init-yoco-path /path/to/YOCO-Llama-8B-Init \
  --global-batch-size 128 \
  --micro-batch-size 1 \
  --lr 5e-5 \
  --warmup-steps 100 \
  --total-steps 8000000 \
  --bf16
```

## Generation / Inference

The baseline supports HF-style `generate()` for the validated single-sequence
mode.

### Minimal Example

```python
import torch
from transformers import AutoTokenizer
from baselines.YOCO.models.YOCO import YOCOForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/path/to/YOCO-Llama-8B-Init")
model = YOCOForCausalLM.from_pretrained("/path/to/YOCO-Llama-8B-Init").eval().cuda()

inputs = tokenizer("Explain YOCO briefly.", return_tensors="pt").to("cuda")
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Current generation constraints:

- batch size `1`
- greedy or otherwise standard single-sequence generation modes
- custom YOCO cache object instead of standard Llama KV cache

## Tensor Parallel Path

[models/YOCO_megatron.py](/data3/junhaohu/comb/baselines/YOCO/models/YOCO_megatron.py)
adapts the YOCO model to the repo's TP helpers.

What it does:

- shards self-decoder attention projections
- shards cross-decoder attention projections
- shards MLP projections
- patches local attention head counts
- gathers the LM head output

What has been validated:

- `TP=1` vs `TP=2` output alignment within expected `bf16` tolerance
- TP cache behavior on decode
- consistent rank losses across TP ranks

## Tests

Available tests:

- [tests/test_yoco_smoke.py](/data3/junhaohu/comb/baselines/YOCO/tests/test_yoco_smoke.py)
- [tests/test_yoco_cache.py](/data3/junhaohu/comb/baselines/YOCO/tests/test_yoco_cache.py)
- [tests/test_yoco_init_from_llama.py](/data3/junhaohu/comb/baselines/YOCO/tests/test_yoco_init_from_llama.py)

Recommended order:

1. run smoke and cache tests
2. run init-from-llama tests
3. run synthetic training smoke
4. run TP smoke

If `pytest` is installed in the environment:

```bash
python -m pytest -q ./baselines/YOCO/tests
```

## Current Verification Status

The baseline has already passed a first round of verification. See
[VERIFICATION_REPORT.md](/data3/junhaohu/comb/baselines/YOCO/VERIFICATION_REPORT.md)
for concrete numbers.

High-level status:

- forward path: verified
- cache semantics: verified
- Llama initialization: verified
- generation: verified
- training viability: verified
- tensor parallel path: verified
- tiny-model comparison vs Llama and Comb: recorded

## Known Limits

The current baseline is intentionally narrow in scope.

Known limits include:

- generation validation is currently for single-sequence mode
- the cache object is custom and not a drop-in replacement for the full HF cache stack
- the training and comparison evidence is still tiny-model and correctness-stage evidence
- this baseline does not yet claim paper-level reproduction or final throughput optimization

## How To Use YOCO As A Baseline

The correct way to use this baseline is:

1. initialize from Llama
2. verify forward / cache / generation on small runs
3. verify tiny-slice overfit with the training script
4. train on the real decoder-only datasets used by the repo
5. compare against:
   - original Llama initialized from the same source
   - current `CombLlama`

That ensures the comparison is about architecture and cache behavior, not about
mismatched preprocessing or mismatched starting points.

## Training And Evaluation Plan For YOCO vs Comb

This is the recommended plan to establish YOCO as a serious baseline in this
repository.

### Phase 1: Baseline Freeze

Goal:

- keep YOCO independent from chunk-based Comb internals
- lock the data interface and layer split

Tasks:

- keep `16 + 16` split fixed
- keep full-parameter training fixed
- do not introduce chunk-encoder dependencies

### Phase 2: Initialization And Sanity

Goal:

- verify that YOCO and Comb start from comparable pretrained sources

Tasks:

- initialize YOCO from `Llama-3.1-8B-Instruct`
- initialize or load the corresponding Comb baseline from the same Llama family
- record parameter counts for all three models:
  - Llama
  - YOCO
  - CombLlama

Outputs:

- init summaries
- parameter table

### Phase 3: Single-Node Training Sanity

Goal:

- prove that YOCO is trainable before large-scale runs

Tasks:

- run synthetic overfit
- run tiny real-data overfit
- verify stable loss decrease
- verify checkpoint save/resume

Match against Comb:

- same optimizer family
- same micro/global batch setup
- same bf16 policy
- same number of update steps

### Phase 4: Real Training Baseline

Goal:

- produce the first real YOCO baseline checkpoint trained under the same repo training stack

Tasks:

- train YOCO with ordinary decoder-only inputs
- train Comb with its chunk-aware inputs
- use the same dataset sequence where possible
- keep learning-rate schedule and total updates aligned

Log for both models:

- train loss
- validation loss
- tokens/sec
- wall-clock time
- checkpoint size
- restart behavior

### Phase 5: Inference Evaluation

Goal:

- compare the practical effect of the architectures during inference

Measure:

- prefill latency
- decode latency
- peak GPU memory
- long-prompt stability
- cache growth behavior

Compare:

- Llama vs YOCO for equal decoder depth and initialization
- YOCO vs Comb for architecture tradeoffs

### Phase 6: Quality Evaluation

Goal:

- determine whether YOCO is only structurally correct or also competitive as a baseline

Evaluate on:

- held-out LM loss / perplexity style metrics if available
- downstream instruction-following evals already used in the repo
- a short curated prompt set for qualitative decode behavior

### Phase 7: Baseline Decision

Decision questions:

- does YOCO train as reliably as Comb?
- does YOCO decode faster or use less memory than Comb or Llama in the target setting?
- does YOCO preserve enough quality to justify its complexity?
- should YOCO remain a standing baseline in future experiments?

### Minimal Comparison Matrix

For every experiment row, keep these columns:

| Model | Init Source | Data Path | Params | Train Loss | Val Loss | Prefill | Decode | Peak Mem | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Llama | Llama-3.1-8B-Instruct | decoder-only | ... | ... | ... | ... | ... | ... | reference |
| YOCO | Llama-3.1-8B-Instruct | decoder-only | ... | ... | ... | ... | ... | ... | pure YOCO |
| CombLlama | Llama-compatible | chunk + decoder | ... | ... | ... | ... | ... | ... | current repo model |

## Practical Recommendation

If you are extending this folder, keep the technical bar high:

- do not reintroduce chunk-model dependencies into YOCO
- do not mark items as complete without execution evidence
- prefer adding tests when fixing cache, generation, or TP logic
- keep README, PLAN, and VERIFICATION_REPORT synchronized with the real state of the code
