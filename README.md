# CombLlama

A hybrid KV cache compression architecture that augments a pretrained Llama decoder with a chunk encoder and cross-attention, reducing memory overhead for long-context LLM inference while preserving generation quality.

## Overview

Standard autoregressive LLMs store full key-value (KV) caches for all prior tokens, leading to memory consumption that grows linearly with context length. CombLlama addresses this by **compressing historical context into compact chunk representations** via a dedicated encoder, then injecting them into the decoder through cross-attention layers. Only recent tokens maintain full KV caches; older context is served through the compressed representations.

### Architecture

```
                    ┌──────────────────────────────────┐
                    │ CombLlamaForConditionalGeneration│
                    └──────────┬───────────┬───────────┘
                               │           │
                  ┌────────────▼──┐   ┌────▼──────────────┐
                  │ Chunk Encoder │   │   Llama Decoder   │
                  │  (8 layers)   │   │   (32 layers)     │
                  │               │   │                   │
                  │ Bidirectional │──►│  Cross-Attention  │
                  │  Self-Attn    │K,V│  at layers        │
                  │  + per-layer  │   │  [3,7,11,15,19,   │
                  │  K/V proj     │   │   23,27,31]       │
                  └───────────────┘   │                   │
                                      │  Causal Self-Attn │
                                      │  + SwiGLU MLP     │
                                      │  (all 32 layers)  │
                                      └───────────────────┘
```

**Chunk Encoder** (`CombLlamaChunkModel`):
- 8 transformer layers with bidirectional self-attention
- Embedding layer shared (copied) from the Llama backbone
- Per-layer K/V projections that produce cross-attention states for each of the 8 decoder cross-attention layers

**Cross-Attention Decoder** (`CombLlamaTextModel`):
- Standard Llama-3.1-8B-Instruct decoder (32 layers) with interleaved cross-attention
- `CombLlamaCrossAttentionDecoderLayer` inserted at 8 configurable layer positions
- Tanh-gated residual connections initialized to zero, preserving pretrained behavior at training start

**Key Design Features**:
- **Variable-length sequence packing**: Uses `flash_attn_varlen_func` with cumulative sequence length tensors (`cu_seqlens`) for padding-free continuous batching
- **Zero-initialized gating**: Cross-attention gates start at zero so the model initially behaves identically to the pretrained backbone
- **Selective training**: Only cross-attention layers and chunk encoder (excluding embedding) are trained; the backbone remains frozen

### Model Specifications

| Component | Value |
|-----------|-------|
| Base Model | Llama-3.1-8B-Instruct |
| Total Parameters | ~11B |
| Trainable Parameters | ~3B (cross-attention + chunk encoder) |
| Backbone Layers | 32 |
| Chunk Encoder Layers | 8 |
| Cross-Attention Positions | [3, 7, 11, 15, 19, 23, 27, 31] |
| Hidden Size | 4096 |
| Attention Heads | 32 (8 KV heads, GQA) |
| Vocabulary Size | 128256 |

## Project Structure

```
comb/
├── models/
│   ├── CombLlama.py              # Model architecture (config, encoder, decoder, LM head)
│   ├── CombLlama_megatron.py     # Tensor parallel adaptation for multi-GPU training
│   └── checkpoint_converter.py   # Bidirectional HF <-> TP checkpoint conversion
├── data/
│   ├── base.py                   # DatasetBase class and collate_fn for varlen packing
│   ├── Ultrachat.py              # UltraChat-200k dataset processor
│   └── distill_data.py           # Dataset distillation utilities
└── training/
    ├── train_llama_megatron.py   # Tensor Parallel (TP) + Data Parallel (DP) training script
    └── plot_training_loss.py     # Training loss visualization
```

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)
- HuggingFace Transformers
- DeepSpeed (for DeepSpeed training) or just PyTorch distributed (for TP training)

```bash
pip install torch transformers datasets flash-attn
```

Access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace is required (gated model).

## Training

CombLlama uses custom tensor parallelism (TP) + optional data parallelism (DP) for distributed training.

```bash
# TP=4, no data parallelism (4 GPUs)
torchrun --nproc_per_node=4 training/train_llama_megatron.py

# TP=2, DP=2 (4 GPUs)
torchrun --nproc_per_node=4 training/train_llama_megatron.py --tp-size 2
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--tp-size` | 4 | Tensor parallel degree |
| `--global-batch-size` | 128 | Global batch size |
| `--lr` | 5e-5 | Learning rate |
| `--warmup-steps` | 100 | Warmup steps |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--bf16` | True | BF16 mixed precision |
| `--steps-per-print` | 10 | Print interval (micro steps) |
| `--resume-ckpt` | None | Checkpoint path to resume from |

**TP size constraint**: Must divide both `num_attention_heads` (32) and `num_key_value_heads` (8). Valid values: 1, 2, 4, 8.

### Checkpoint Conversion

Convert between HuggingFace and TP-sharded formats:

```bash
# HuggingFace -> TP shards
python models/checkpoint_converter.py hf2tp \
    --hf-path /path/to/hf_model \
    --output-dir /path/to/tp_shards \
    --tp-size 4

# TP shards -> HuggingFace
python models/checkpoint_converter.py tp2hf \
    --tp-dir /path/to/tp_checkpoint \
    --output-dir /path/to/hf_model \
    --tp-size 4
```

## Usage

### Inference

```python
from transformers import LlamaConfig
from models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration

# Load trained model
model = CombLlamaForConditionalGeneration.from_pretrained("/path/to/checkpoint")
model.eval().cuda()

# Or initialize from scratch (loads pretrained Llama backbone)
config = CombLlamaConfig(LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"))
model = CombLlamaForConditionalGeneration(config=config, from_scratch=True)
```

### Data Format

The model expects packed variable-length inputs with cumulative sequence length tensors:

```python
# input_ids:       [1, total_input_len]     — decoder token ids
# chunk_ids:       [1, total_chunk_len]     — chunk encoder token ids
# cu_seqlens_q:    [batch + 1]             — decoder sequence boundaries
# cu_seqlens_k:    [batch + 1]             — per-sample total chunk length boundaries
# cu_seqlens_chunk:[num_chunks + 1]        — individual chunk boundaries
# position_ids:    [1, total_input_len]     — decoder position ids
# position_ids_k:  [1, total_chunk_len]     — chunk encoder position ids
# shift_labels:    [1, total_input_len]     — pre-shifted labels (-100 for masked positions)
```

The `collate_fn` in `data/base.py` handles this packing automatically from dataset samples.

## How It Works

During inference with long conversations:

1. **Older dialogue turns** are grouped into chunks (up to 1024 tokens each) and processed by the chunk encoder into compressed K/V representations
2. **The decoder** attends to these compressed representations via cross-attention at 8 interleaved layers, while performing standard causal self-attention on recent tokens
3. **Memory savings** come from replacing full KV caches for historical tokens with the compact chunk encoder output

This is analogous to how humans process long conversations — retaining the gist of earlier exchanges while maintaining detailed memory of recent context.

## License

Apache License 2.0
