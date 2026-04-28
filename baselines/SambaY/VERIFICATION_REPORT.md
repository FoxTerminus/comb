# SambaY Verification Report

Date: 2026-04-28

## Scope

This report closes Stage 8 for the `SambaY-Llama-8B-Init` baseline. It records
the checks that make failures local before benchmark integration.

Primary checkpoint:

```text
/data3/junhaohu/model/SambaY-Llama-8B-Init
```

Backbone checkpoint:

```text
/data3/junhaohu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
```

GPU policy:

```text
CUDA_VISIBLE_DEVICES=5,6
```

No verification command in this stage intentionally used physical GPU0 or GPU1.

## Architecture

The implementation follows the SambaY paper and `microsoft/ArchScale` rather
than a naive Samba + YOCO composition.

The 32-layer Llama-3.1-8B backbone is mapped to the ArchScale `sambay_d*`
schedule:

- layers `0..15`: Samba local self-decoder layers
- layer `16`: forced Mamba-1 `gmu_save` layer
- layer `17`: full-attention boundary layer that creates the shared YOCO K/V
- layers `18..31`: GMU and shared-KV cross-attention layers

Verified architectural properties:

- `d_model = 4096`
- `d_mem = 8192`
- `rnn_per_layer = 2`
- `gmu_per_layer = 2`
- NoPE is enabled for the official baseline path
- cross-attention layers own Q/O only and reuse the single boundary K/V cache
- GMU layers consume Mamba SSM scan memory, not Comb chunk states
- CUDA Mamba prefill uses ArchScale `gmu_save=True` semantics: scan output before
  SiLU gating and before `out_proj`

Detailed architecture audit:

```text
baselines/SambaY/ARCHITECTURE_AUDIT.md
```

## Initialization

The initialized checkpoint was audited after running the Llama-to-SambaY script.

Summary from `init_summary.json`:

- `missing_keys`: `[]`
- `unexpected_keys`: `[]`
- dtype: `torch.bfloat16`
- total parameters: `9,925,951,488`
- trainable parameters: `9,925,951,488`
- checkpoint size: `37G`
- GMU memory size: `8192`

Spot-checked exact Llama weight copies:

- embeddings and LM head
- self-decoder SWA Q/K/V/O from Llama layer 1
- forced `gmu_save` layer MLP from Llama layer 16
- boundary full-attention Q/K/V from Llama layer 17
- cross-attention Q/O from Llama layer 19

No copied-parameter record targets the forced Mamba mixer or GMU modules.

Detailed initialization audit:

```text
baselines/SambaY/POST_INIT_AUDIT.md
```

## Unit Tests

Command:

```bash
PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/mamba/bin/python \
  -m pytest baselines/SambaY/tests -q
```

Result:

```text
17 passed
```

Coverage:

- `test_sambay_smoke.py`: tiny forward/loss, layer layout, GMU/cross-attn
  interleaving
- `test_sambay_gmu.py`: GMU shape and gradient flow
- `test_sambay_cache.py`: prefill/decode consistency and greedy generation
- `test_sambay_data.py`: SambaY collate and ordinary LM label contract
- `test_sambay_native_data.py`: raw multi-turn chat conversion and assistant
  supervision
- `test_sambay_init_from_llama.py`: tiny Llama initialization, reload, schedule
  validation, and rejection of invalid layer counts
- `test_sambay_training.py`: synthetic tiny training loss decrease and checkpoint
  resume
- `test_sambay_megatron.py`: TP adapter world-size-one path and gated-pair
  projection layout

## Distributed TP Smoke

Command:

```bash
PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/mamba/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  baselines/SambaY/tests/distributed_tp2_smoke.py
```

Observed result:

```text
tp2_loss_avg=4.964398
tp2_generate_shape=(1, 5)
```

Verified:

- TP=2 forward/backward runs
- rank losses match within test tolerance
- cache-enabled generation runs under TP for batch size 1

## Full Checkpoint CUDA Smoke

Forward smoke:

```text
visible device 0: NVIDIA A100-SXM4-80GB
loss: 14.586273193359375
logits shape: (1, 4, 128256)
shared K shape: (1, 4, 8, 128)
GMU memory shape: (1, 4, 8192)
peak memory: 18.503 GiB
```

Generation smoke:

```text
uses_archscale_mamba_gmu_save_layer: True
generated shape: (1, 5)
peak memory: 18.502 GiB
```

These checks loaded `/data3/junhaohu/model/SambaY-Llama-8B-Init` directly and
ran on `CUDA_VISIBLE_DEVICES=5,6`.

## Training Smoke

CUDA policy:

```text
CUDA_VISIBLE_DEVICES=5,6
```

Tiny real-data training smoke:

```text
model: tiny SambaY config with Llama tokenizer vocab
dataset: ultrachat_200k
steps: 2
max_train_seq_len: 128
micro_batch_size: 1
device: cuda
```

Logs:

```text
/data3/junhaohu/tmp/sambay_real_smoke_logs/training_loss.csv
/data3/junhaohu/tmp/sambay_real_smoke_logs/training_diagnostics.csv
```

Observed losses:

```text
step 1: 11.767436981201172
step 2: 11.766627311706543
```

Checkpoint:

```text
/data3/junhaohu/tmp/sambay_real_smoke_ckpt/step_2/model.pt
```

Full initialized checkpoint backward-only smoke:

```text
checkpoint: /data3/junhaohu/model/SambaY-Llama-8B-Init
loss: 15.65579605102539
lm_head grad finite: True
embedding grad finite: True
GMU grad finite: True
boundary attention Q grad finite: True
peak memory: 37.199 GiB
```

The full-checkpoint smoke intentionally did not run an AdamW optimizer step.
Single-process AdamW for the 9.93B SambaY checkpoint would allocate optimizer
states that are expected to exceed one 80GB A100. Formal full-checkpoint
training should use the tensor-parallel/distributed path or an optimizer state
partitioning strategy.

## Known Limitations

- CPU tests use the deterministic fallback Mamba recurrence so CI-like checks do
  not depend on CUDA kernels.
- Tensor-parallel Mamba uses the fallback recurrence because the installed
  official Mamba kernels are not sharded by the current local TP adapter.
- CUDA token-by-token Mamba state update uses the correctness path. Prefill uses
  the ArchScale scan-memory path.
- Beam cache reorder is documented and limited to batch size 1.
- The baseline has not yet been benchmarked against Llama, YOCO, or CombLlama.

## Stage 8 Status

Stage 8 is complete.

All planned tests exist and pass, the full initialized checkpoint has been
audited, and the remaining work belongs to Stage 9 benchmark integration.
