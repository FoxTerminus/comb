# Comb 研究实施计划（Qwen3-0.6B backbone — REVISION 1）

**日期**: 2026-05-01  
**目标**: 以 Qwen3-0.6B 为 backbone，构建 Comb baseline，在 ProLong-64K 数据上训练，参数量与 SambaY-1B（~1.13B）对齐，用于公平对比。

---

## 用户约束

- **命名**: 新代码统一使用 `Comb`，不再使用 `CombLlama` 作为新 API 名称。历史 `archive/CombLlama.py` 保留为参考。
- **cross-attention 层数**: 保持 backbone 层数的 **1/4**。28 层 → **7 个** cross-attention 层。
- **encoder 同源初始化**: encoder 和 decoder 使用同一个 backbone 模型家族，encoder 层参数从 decoder backbone 对应层复制初始化。
- **GPU 固定策略**: 4 卡 `2,3,4,7`；2 卡 `5,6`；单卡 `6`。

---

## 1. Qwen3-0.6B 官方配置

来源: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json

```python
Qwen3Config(
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,         # GQA
    head_dim=128,                  # 不是 1024/16=64!
    intermediate_size=3072,
    vocab_size=151936,
    max_position_embeddings=40960,
    tie_word_embeddings=True,      # embed 与 lm_head 共享权重
    rms_norm_eps=1e-6,
)
```

**关键**: `head_dim=128` 意味着 attention projection 维度为 `q_proj: 1024→16×128=2048`，`k_proj: 1024→8×128=1024`，不是传统 Llama 的 `hidden/n_heads`。

---

## 2. 参数量估算

按 Qwen3-0.6B `head_dim=128` 和 7 层 encoder/cross-attention：

| 组件 | 估算 |
|---|---|
| Qwen3-0.6B backbone（28 层, tied embed） | ~606M |
| Chunk embedding（复制 backbone embed，独立 Parameter） | ~155M |
| 7 层 encoder（attention+MLP, 每层 ~10.5M） | ~74M |
| 7 层 encoder per-layer K/V projection（2 × 1024×1024 per layer） | ~15M |
| 7 个 cross-attn Q/O proj（2 × 1024×2048 per layer） | ~27M |
| 7 个 cross-attn gated MLP + norms | ~35M |
| **总参数** | **~915M**（待实测） |

> 展望: 预期实测 **~900M-1.0B**，与 SambaY-1B（1.13B）可比。精确值用 `tools/inspect_comb.py` 实测确认。

---

## 3. 代码结构规划

```
./comb/
├── models/
│   ├── __init__.py
│   ├── comb_qwen.py             # Qwen3 Comb 实现
│   ├── comb_qwen.py             # 新 Comb 实现（基于 Qwen3）
│   └── config.py                # CombConfig（独立文件）
├── data/
│   ├── __init__.py
│   └── prolong_qwen_dataset.py  # ProLong-64K Qwen3-tokenized LITPKDS 数据加载器
├── training/
│   ├── __init__.py
│   └── train.py                 # 分布式训练脚本（FSDP2）
├── configs/
│   ├── comb_qwen_1b.yaml        # 完整配置（7 cross-attn layers）
│   └── comb_tiny.yaml           # tiny smoke test 配置
├── tools/
│   └── inspect_comb.py          # 拓扑 + 参数量打印
├── scripts/
│   ├── tokenize_prolong_qwen.py # Llama2 LITPKDS → decode → Qwen3 LITPKDS
│   └── run_comb_train.sh        # 正式训练启动脚本
├── PLAN.md
├── REPORT.md
└── REVIEW.md
```

---

## 4. 模型适配方案

### 4.1 Llama → Qwen3 组件替换

| 原 CombLlama 组件 | Qwen3 等效 |
|---|---|
| `LlamaConfig` | `Qwen3Config` |
| `LlamaDecoderLayer` | `Qwen3DecoderLayer` |
| `LlamaRMSNorm` | `Qwen3RMSNorm` |
| `LlamaMLP` | `Qwen3MLP` |
| `LlamaRotaryEmbedding` | `Qwen3RotaryEmbedding` |
| `apply_rotary_pos_emb` | `apply_rotary_pos_emb`（Qwen3 版本） |

### 4.2 Cross-Attention 插入策略（7 层）

```python
cross_attention_layers = [3, 7, 11, 15, 19, 23, 27]
# 28 / 4 = 7 层，间隔 4 层，风格对齐原 CombLlama 的 [3,7,11,15,19,23,27,31]
```

### 4.3 Encoder 同源初始化策略

基于用户约束「encoder 和 decoder 使用同一个 backbone」：

1. **Chunk embedding**: 从 Qwen3 backbone `model.embed_tokens` 复制权重，**冻结**
2. **Chunk encoder 层** (7 层): 从 Qwen3 decoder 的对应层复制注意力/MLP/norm 参数：
   - encoder layer[i] ← decoder layer[cross_attention_layers[i]]
   - 例如 encoder layer 0 ← decoder layer 3, encoder layer 1 ← decoder layer 7, ...
3. **Per-layer K/V projection**: 从对应 decoder self-attention 的 `k_proj`/`v_proj` 复制初始化
4. **Cross-attention Q/O proj**: 从对应 decoder self-attention 的 `q_proj`/`o_proj` 复制初始化
5. **Cross-attention gates**: **zero-initialized**（保持预训练行为）
6. **冻结策略**（与原 CombLlama `from_scratch=True` 一致）:

**冻结**:
- decoder/backbone embedding
- decoder/backbone 全部 self-attention + MLP 层
- decoder final norm + LM head
- encoder/chunk embedding

**可训练**:
- encoder/chunk 7 层 attention/MLP/norm 主体
- encoder/chunk per-layer `k_proj` / `v_proj`
- decoder 侧新增 cross-attention layers 全部参数（`q_proj`、`o_proj`、norm、gates、gated MLP）

### 4.4 关键 shape 验证

所有 attention 实现必须验证：
- Q shape: `[total_len, num_attention_heads=16, head_dim=128]`
- K/V shape: `[total_len, num_key_value_heads=8, head_dim=128]`
- Q proj output: `num_heads × head_dim = 16 × 128 = 2048`
- KV proj output: `num_kv_heads × head_dim = 8 × 128 = 1024`
- Cross-attn Q proj: `1024 → 2048`
- Cross-attn K/V 来自 encoder `k_proj[idx]`: `1024 → 1024`

---

## 5. 数据方案

### 5.1 ProLong 数据策略

**当前推荐工程路线**:
```
Llama2-tokenized LITPKDS → Llama2 tokenizer decode → text → Qwen3 tokenizer → Qwen3-tokenized LITPKDS
```

**备注**: `princeton-nlp/prolong-data-64K` 可访问，但它是 Llama-3 tokenized MDS（`input_ids`，非 raw text）。MDS 原始读取也可作为备选路线，但需要 `mosaicml-streaming`。

### 5.2 Tokenize 方案

```
Llama2 LITPKDS → decode → text → Qwen3 tokenizer → new LITPKDS .bin → ProLongQwenDataset
```

与 `baselines/ArchScale/data/prolong_dataset.py` 保持相同的 LITPKDS 格式。

### 5.3 数据契约

- `Dataset.__getitem__` 只返回 raw 64K block: `{"input_ids": tensor[65536], "labels": tensor[65536]}`
- `labels == input_ids`（模型内部做 shift 或 collate_fn 处理）
- `collate_fn` 负责构造 Comb 训练所需的 `chunk_ids`、`cu_seqlens_q/k/chunk`、`position_ids` 等
- Pretraining 场景所有有效 token 都参与 next-token loss，`shift_labels[t] = input_ids[t+1]`，末位设为 `-100`（不做 user/assistant masking，那是 SFT 的逻辑）
- 只使用 train split
- 子采样 1/20
- `block_size=65536`

### 5.4 Chunk IDs 构造

Pretraining 场景简化方案：每个 64K block 切成等长 chunks（如 64 chunks × 1024 tokens），chunk encoder 处理所有 chunks 后产生 K/V 给 cross-attention 使用。具体实现由 `collate_fn` 完成。

---

## 6. 训练方案

### 6.1 超参数（对齐 SambaY）

| 参数 | 值 |
|---|---|
| ctx_len | 65536 |
| peak lr | 3e-4 |
| min lr | 3e-5 |
| warmup_steps | 591 (10%) |
| total_steps | 5913 |
| micro_bsz | 1 |
| grad_accum | 1 |
| optimizer | AdamW (β1=0.9, β2=0.95) |
| weight_decay | 0.1 |
| precision | bf16 autocast |
| activation ckpt | True（64K 必需） |
| seed | 42 |

### 6.2 训练脚本

参考 `baselines/ArchScale/training/train.py`：
- FSDP2（顶层 `fully_shard`）
- Cosine LR
- GPU guard（按阶段校验: 单卡只允许 6，2卡只允许 5,6，4卡只允许 2,3,4,7；不设全局"禁用5"）
- `--act-ckpt`
- Checkpoint save/resume (`--resume auto`)
- `training_diagnostics.csv`（global_step, loss, lr），**不含 grad_norm**

### 6.3 GPU 分配

| 阶段 | GPU |
|---|---|
| 单卡 smoke | `CUDA_VISIBLE_DEVICES=6` |
| 2 卡 smoke | `CUDA_VISIBLE_DEVICES=5,6` |
| 4 卡 64K preflight | `CUDA_VISIBLE_DEVICES=2,3,4,7` |
| 4 卡正式训练 | `CUDA_VISIBLE_DEVICES=2,3,4,7` |

---

## 7. 实施步骤

### 步骤 0: 配置验证与数据可用性

**产出写入 `REPORT.md`，review 通过后进入步骤 1。**

1. Qwen3-0.6B 官方 config 读取（确认 head_dim=128, hidden=1024, 28 layers, tie_word_embeddings=True）
2. **64K position/RoPE 可行性验证**:
   - Qwen3 `max_position_embeddings=40960` → 训练 `ctx_len=65536`，确认 RoPE 在 position_ids 到 65535 时是否能正常 forward
   - 是否需把 `max_position_embeddings` 改为 65536
   - 是否需要 rope scaling / YaRN / 仅扩表
3. 参数量公式估算（步骤 0）→ `inspect_comb.py` 实测（步骤 2 后补充）:
   - 用公式和 Qwen3 backbone 参数读取结果估算 total/trainable/frozen
   - `REPORT.md` 明确哪些是估算、哪些待实测
4. ProLong 数据可用性验证:
   - 仓库 `princeton-nlp/prolong-data-64K` 可访问（Llama-3 tokenized MDS，非 raw text）
   - 已有 LITPKDS（Llama2-tokenized）可作为工程路线起点
   - 能否只抽取 train split 子集

### 步骤 1: 代码骨架

- 创建 `models/__init__.py`、`data/`、`training/`、`configs/`、`tools/`、`scripts/`
- 旧 `CombLlama.py` 已归档到 `archive/CombLlama.py`
- 创建空白 `comb_qwen.py` 和 `config.py`

### 步骤 2: 模型适配

- 实现 `models/comb_qwen.py`：CombConfig + CombChunkModel + CombTextModel + CombForCausalLM
- 7 层 encoder（cross_attention_layers = [3,7,11,15,19,23,27]）
- Encoder 同源初始化（embedding + 层参数 + K/V proj 从 backbone 复制）
- 实现 `tools/inspect_comb.py`
- 验证: CPU tiny forward/backward + GPU tiny smoke + 参数量实测

### 步骤 3: 数据管线

- Llama2 LITPKDS decode → Qwen3 tokenizer → new LITPKDS .bin
- 子采样 1/20 train-only
- 实现 `data/prolong_qwen_dataset.py`
- 验证: header/token/labels/determinism

### 步骤 4: 训练脚本

- 实现 `training/train.py`
- 创建 `configs/comb_qwen_1b.yaml` 和 `configs/comb_tiny.yaml`

### 步骤 5: Smoke test + 64K preflight

- Tiny config 单/双 GPU smoke
- 64K preflight (GPU 2,3,4,7)

### 步骤 6: 正式训练

---

## 8. 验收清单

- [ ] Qwen3-0.6B `head_dim=128` 确认
- [ ] 64K position/RoPE 可行性确认（position_ids 到 65535 正常 forward）
- [ ] 参数量实测（total/trainable/frozen + per-component）~900M-1.0B
- [ ] cross_attention_layers = `[3,7,11,15,19,23,27]` (7 layers)
- [ ] Encoder 参数从 decoder backbone 复制初始化
- [ ] 训练/冻结边界与原 CombLlama 对齐: backbone 冻结, encoder 主体可训练, cross-attn 可训练
- [ ] GPU guard 按阶段校验（单卡 6, 2卡 5/6, 4卡 2/3/4/7），不设全局"禁用5"
- [ ] 新 API 命名统一为 `Comb`（非 CombLlama）
- [ ] ProLong 数据可用性确认（步骤 0: LITPKDS decode→re-tokenize 路线可行）
- [ ] Tiny smoke: forward/backward + 2 steps
- [ ] 64K preflight: loss 下降, peak mem < 80GB
- [ ] `training_diagnostics.csv` 写入 global_step/loss/lr
- [ ] 正式训练命令可复现
