# SambaY-Llama Stage 0 Design

## Boundary

`SambaY-Llama-8B-Init` is a standalone causal-LM baseline initialized from
`Llama-3.1-8B-Instruct`. It uses the same raw datasets as the other baselines,
but owns its model, cache, preprocessing, collate function, and training path.

It must not use `CombLlama` chunk fields such as `chunk_ids`, `chunk_model`, or
`cross_attention_states`.

## Architecture

The target architecture follows the SambaY paper and `microsoft/ArchScale`:

- self-decoder: Samba-style hybrid stack following ArchScale global layer indexing
- local token mixing: Mamba-1 and sliding-window attention
- GMU memory producer: one forced Mamba-1 `gmu_save` layer at the YOCO boundary
- KV memory producer: one following full-attention layer that produces a shared KV cache
- cross-decoder: interleaved GMU and cross-attention layers after the GMU/KV producer layers
- GMU memory: Mamba SSM scan output before SiLU gating and output projection, with `d_mem = 2 * d_model`
- positional encoding: NoPE by default; RoPE is only an ablation flag

## Llama Initialization

Directly copy compatible Llama modules:

- embeddings
- final norm
- LM head
- RMSNorm and MLP blocks where layer dimensions match
- Q/O and shared K/V attention projections where structurally meaningful

Mamba and GMU modules are newly initialized and trained with all parameters.
The forced boundary `gmu_save` layer copies only compatible Llama RMSNorm/MLP
weights; its Mamba mixer has no Llama analogue.

## Data Contract

SambaY has a dedicated preprocessing path:

- `preprocess_sambay_example(...)`
- `collate_fn_sambay(...)`

The collated batch contains ordinary decoder tokens and SambaY-specific packed
sequence metadata. It does not contain chunk fields.
