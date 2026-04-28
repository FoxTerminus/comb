# SambaY Post-Initialization Audit

Date: 2026-04-28

Checkpoint:

```text
/data3/junhaohu/model/SambaY-Llama-8B-Init
```

Backbone:

```text
/data3/junhaohu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
```

## Summary Checks

- `missing_keys`: `[]`
- `unexpected_keys`: `[]`
- dtype: `torch.bfloat16`
- total/trainable parameters: `9,925,951,488 / 9,925,951,488`
- checkpoint size: `37G`
- NoPE: `True`
- GMU memory size: `8192`

ArchScale schedule recorded by `init_summary.json`:

- self-decoder local layers: `[0, 15]`
- forced Mamba `gmu_save` layer: `16`
- boundary full-attention shared-KV layer: `17`
- cross-decoder layers: `[18, 31]`
- cross-decoder global layer indices: `[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]`
- cross-decoder GMU pattern: `[True, False, True, False, True, False, True, False, True, False, True, False, True, False]`

## Weight-Copy Spot Checks

The following initialized SambaY tensors were exactly equal to their intended
Llama source tensors:

- embeddings and LM head
- self-decoder SWA Q/K/V/O from Llama layer 1
- forced `gmu_save` layer MLP from Llama layer 16
- boundary full-attention Q/K/V from Llama layer 17
- cross-attention Q/O from Llama layer 19

No copied-parameter entry incorrectly targets the forced Mamba mixer or GMU
modules.

## Runtime Checks

All CUDA checks used:

```text
CUDA_VISIBLE_DEVICES=5,6
```

Full checkpoint forward:

- visible CUDA device 0 maps to `NVIDIA A100-SXM4-80GB`
- finite loss: `14.586273193359375`
- logits shape: `(1, 4, 128256)`
- shared K shape: `(1, 4, 8, 128)`
- GMU memory shape: `(1, 4, 8192)`
- peak memory: `18.503 GiB`

Full checkpoint generation:

- `uses_archscale_mamba_gmu_save_layer`: `True`
- generated shape: `(1, 5)`
- peak memory: `18.502 GiB`

Unit tests:

```text
pytest baselines/SambaY/tests -q
```

Result:

```text
17 passed
```

## Notes

- The checkpoint is larger than Llama-3.1-8B because SambaY adds Mamba and GMU
  parameters. The current initialized model is about 9.93B parameters.
- The generation smoke emitted the expected Transformers warning about missing
  `attention_mask`; the shape/cache path still executed successfully. Training
  and benchmark scripts should provide masks or packed sequence metadata.
