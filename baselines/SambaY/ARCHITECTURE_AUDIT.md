# SambaY Architecture Audit

Date: 2026-04-28

## References Checked

- SambaY paper: `arXiv:2507.06607`
- Official source: `microsoft/ArchScale`
  - `lit_gpt/config.py`
  - `lit_gpt/model.py`
  - `lit_gpt/mamba_simple.py`
  - `lit_gpt/gated_memory_unit.py`
  - `lit_gpt/attention.py`

## Confirmed Alignment

- The model is a standalone causal LM baseline and does not consume Comb chunk fields.
- The Llama-3.1-8B backbone config is preserved for vocab size, hidden size, MLP width, layer count, attention heads, KV heads, RMSNorm eps, and LM head shape.
- The default split follows the ArchScale `sambay_d*` global schedule for a 32-layer backbone:
  - layers `0..15`: Samba local self-decoder layers
  - layer `16`: forced Mamba-1 `gmu_save` layer
  - layer `17`: full-attention shared-KV producer
  - layers `18..31`: GMU/cross-attention layers
- Mamba-1 GMU memory now uses the ArchScale `gmu_save=True` semantics on CUDA: the memory is the selective-scan output before SiLU gating and before `out_proj`.
- GMU implements `out_proj(SiLU(in_proj(hidden)) * memory)`.
- Cross-attention layers own Q/O only; K/V come from the single shared boundary KV.
- NoPE remains the official default.
- Llama initialization copies compatible embeddings, final norm, LM head, RMSNorm/MLP, and attention projections. Mamba/GMU modules remain newly initialized.

## Intentional Limitations

- CPU tests use a deterministic fallback Mamba recurrence so architecture/data/cache tests remain runnable without CUDA kernels.
- Tensor-parallel Mamba uses the fallback recurrence because the installed official Mamba kernels are not sharded by this local TP adapter.
- CUDA token-by-token Mamba state update still uses the correctness path rather than a fused vLLM-style state update. Prefill uses the ArchScale scan-memory path.
- Full `meta-llama/Llama-3.1-8B-Instruct` initialization is not yet executed in this workspace; the script has been validated on a tiny local Llama fixture.

## Verification Run

- `pytest baselines/SambaY/tests -q`: 16 passed.
- `torch.distributed.run --nproc_per_node=2 baselines/SambaY/tests/distributed_tp2_smoke.py`: finite averaged TP2 loss and cache generation shape `(1, 5)`.
- CUDA smoke with `CUDA_VISIBLE_DEVICES=5,6`: finite loss, visible devices are two A100s, GMU memory shape is `(batch, seq, 2 * hidden)`.
