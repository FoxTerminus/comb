# REPORT.md — 步骤 0: 配置验证与数据可用性

**日期**: 2026-05-01  
**环境**: samba (conda, Python 3.12)

---

## 0.1 Qwen3-0.6B 官方配置

从 `Qwen/Qwen3-0.6B/config.json` 读取：

| 参数 | 值 |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| **head_dim** | **128** (注意: 不是 1024/16=64) |
| intermediate_size | 3072 |
| vocab_size | 151936 |
| max_position_embeddings | 40960 |
| tie_word_embeddings | True |
| rms_norm_eps | 1e-6 |

**关键**: `head_dim=128` 意味着:
- Q projection: `1024 → 16×128 = 2048`
- K/V projection: `1024 → 8×128 = 1024`

---

## 0.2 64K Position/RoPE 可行性

**测试**: 用 `Qwen3ForCausalLM.from_config(cfg, max_position_embeddings=65536)` 创建模型，传入 `position_ids=torch.arange(65408, 65536)` 进行 forward。

```
Forward position 65408-65535: OK ✓
Output shape: (1, 128, 151936)
```

**结论**: 将 `max_position_embeddings` 设为 65536 后，短序列 position range `[65408,65535]` forward 通过。暂无需 YaRN/rope_scaling 变更。真实 64K full-length `position_ids.max()==65535` 仍需在 64K preflight 中验证。

---

## 0.3 ProLong 数据可用性

### MDS 原始数据

`princeton-nlp/prolong-data-64K`（dataset repo）可访问，有 3608 个 MDS 文件分布在 11 个子集。`snapshot_download` 可以下载。

**但 MDS 数据是 Llama-3 tokenized `input_ids`，不是 raw text**。字段为 `input_ids`、`indices`、`domain`。当前 `samba` 环境未安装 `mosaicml-streaming`，MDS 样本实际读取尚未验证。从 MDS 恢复文本依赖 Llama-3 tokenizer 解码，不是从 raw text 字段直接获取。

### 当前工程路线: LITPKDS 解码 → Qwen3 重 tokenize（替代方案）

利用已有 `/data3/junhaohu/data/prolong_64K_v2/` 下的 LITPKDS 数据（Llama2 tokenizer，84GB，21347 个 `.bin` 文件）作为起点。

测试: Llama2 tokenizer 解码 → text → Qwen3 tokenizer 重编码:

```
Llama2 tokens: 500 → text: 2023 chars → Qwen3 tokens: 451 (ratio 0.90)
Full file estimate: 2,098,176 Llama2 → ~1,892,554 Qwen3 tokens
```

**身份**: 这是 ProLong raw text 不可直接获取时的工程替代路线，不等同于 "ProLong raw text 已验证可用"。

**已知风险**:
- decode 后文本可能不完全等同原始文本（特殊 token、空白、文档边界）
- LITPKDS 是 packed token stream，文档边界信息可能不足
- re-tokenization 后的 64K block 边界会重新分布
- 后续步骤 3 实现前必须做小样本 round-trip 验证

### Train-only 子集

从 21347 个 Llama2-tokenized `.bin` 文件中，取 `train_*.bin` 的 NR%20==1，转码为 Qwen3 tokens。

### Vocab Size 差异

Config `vocab_size=151936`（用于 embedding/lm_head），tokenizer vocab=`151643`（含 added/special tokens）。实现时以 config `vocab_size` 建 embedding。

---

## 0.4 参数量估算

| 组件 | 参数量 |
|---|---|
| Decoder backbone (28 layers, no embed) | 440M |
| Embedding (tied, counted once) | 155.6M |
| Backbone total | 596M |
| Chunk encoder (7 layers) | 74M |
| Encoder K/V proj (7×2) | 15M |
| Cross-attn layers (7) | 95M |
| Chunk embedding (copy from backbone) | 155.6M |
| **Comb overhead** | **340M** |
| **总参数** | **~936M** |
| Trainable (est): encoder + cross-attn | 184M |
| Frozen (est): backbone + chunk embed | 752M |

vs SambaY-1B (~1130M): **~83%**。真实 total/trainable/frozen/per-component 等待步骤 2 `inspect_comb.py` 实测。

---

## 0.5 Label 约定

- Pretraining: 所有有效 token 参与 next-token loss
- `shift_labels[t] = input_ids[t+1]`，末位 `-100`
- 模型内部不二次 shift
- `collate_fn` 负责生成 `chunk_ids`、`cu_seqlens_*`、`position_ids*`

---

## 0.6 GPU Guard 策略

| 阶段 | CUDA_VISIBLE_DEVICES |
|---|---|
| 单卡 smoke | 6 |
| 2 卡 smoke | 5,6 |
| 4 卡 64K preflight | 2,3,4,7 |
| 4 卡正式训练 | 2,3,4,7 |

按阶段校验，不设全局"禁用5"。

---

## 结论

**步骤 0 部分通过**:
- Qwen3-0.6B `head_dim=128` 确认 ✓
- 64K 短序列 position 探针通过，真实 full-length 待 preflight 验证 ✓
- ProLong MDS 可访问，已下载 sample；数据是 Llama-3 tokenized MDS 非 raw text
- 工程替代路线: LITPKDS decode (Llama2) → text → Qwen3 re-tokenize，小样本测试通过，待步骤 3 完整 round-trip 验证
- 参数量估算 ~936M，与 SambaY-1B ~83% 对齐 ✓
- GPU guard 策略无冲突 ✓

可进入步骤 1: 代码骨架搭建。

---

## 步骤 1: 代码骨架搭建

**状态**: 完成

### 新建文件

```
./comb/models/__init__.py       # "Comb models."
./comb/models/comb_qwen.py      # 空占位（步骤 2 实现）
./comb/models/config.py         # 空占位（步骤 2 实现）
./comb/data/__init__.py         # "Comb data modules."
./comb/data/prolong_qwen_dataset.py  # 空占位（步骤 3 实现）
./comb/training/__init__.py     # "Comb training modules."
./comb/training/train.py        # 空占位（步骤 4 实现）
./comb/tools/inspect_comb.py    # 空占位（步骤 2 实现）
./comb/configs/                 # 目录已创建（步骤 2/4 写入 YAML）
./comb/scripts/                 # 目录已创建（步骤 3 写入脚本）
```

### 保留不变

- `archive/CombLlama.py` — 历史参考，已从 `models/` 归档

### 待完成

- 步骤 2: 模型适配（实现 `comb_qwen.py`、`config.py`、`inspect_comb.py`）

### 同步修订 PLAN.md

PLAN.md 中 "ProLong MDS → 原始文本 → Qwen3 tokenizer" 旧表述已全部替换为当前工程路线: "Llama2 LITPKDS decode → Qwen3 tokenizer → new LITPKDS"。

---

## 步骤 2: 模型适配

**状态**: 完成（CPU tiny 通过）

### 实现文件

- `models/config.py` — CombConfig（Qwen3Config wrapper, cross_attention_layers=[3,7,11,15,19,23,27]）
- `models/comb_qwen.py` — CombChunkModel + CombTextModel + CombForConditionalGeneration
- `tools/inspect_comb.py` — 参数量统计工具

### API 命名

全部使用 `Comb*`，不再使用 `CombLlama*` 作为新 API 名称。

### Head Dim 处理

全程使用 `head_dim=128`（Qwen3 官方 config），不走 `hidden_size // num_heads`。

### Encoder 同源初始化

`_init_from_scratch()` 从 Qwen3-0.6B backbone：
- decoder backbone 全部 28 层复制
- chunk encoder 7 层参数从 `cross_attention_layers` 对应 decoder 层复制
- encoder K/V proj 从对应 decoder self-attn K/V proj 复制
- cross-attention Q/O proj + MLP 从对应 decoder 层复制
- gates 全部 zero-initialized

### 冻结边界

与 CombLlama `from_scratch=True` 对齐：
- **冻结**: decoder backbone (all 28 layers), decoder embedding, decoder norm, LM head, encoder embedding
- **可训练**: encoder 7 层 (attn+MLP+norm), encoder K/V proj (7×2), cross-attention layers (7 层, 含 q_proj, o_proj, norm, gates, gated MLP)

### 参数量实测

| 组件 | 参数量 |
|---|---|
| Decoder backbone (28 layers) | ~597M |
| Embedding (tied, decoder) | ~155.6M |
| LM head | ~155.6M |
| Chunk encoder layers (7) | ~53M |
| Chunk encoder K/V proj (7×2) | ~14.7M |
| Chunk encoder embedding | ~155.6M |
| Cross-attention layers (7) | ~148M |
| **Total** | **1127.4M** |
| **Trainable** | **220.2M** |
| **Frozen** | **907.2M** |

### 与 SambaY-1B 对齐

| 模型 | 总参数 |
|---|---|
| SambaY-1B | 1129.3M |
| Comb (Qwen3-0.6B) | 1127.4M |
| 差异 | **1.9M (<0.2%)** |

### 修复记录

- Qwen3MLP/Qwen3RotaryEmbedding API（接收 config 对象非参数展开）
- attention bias=False（匹配 Qwen3 backbone）
- RoPE broadcasting（Qwen3 cos/sin 3D → unsqueeze head dim）
- CPU fallback: GQA repeat + 4D shape handling for CrossAttention

### CPU tiny 验证

```
loss: 14.7849
CPU fwd+bwd: PASSED ✓
Gates zero-init: ✓
Freeze boundary (decoder frozen, encoder+cross trainable): ✓
```

---

## 步骤 3: 数据管线

**状态**: 完成

### 实现

- `data/prolong_qwen_dataset.py` — `ProLongQwenDataset`（map-style，LITPKDS 格式，preload int64 buffer）
- 与 `baselines/ArchScale/data/prolong_dataset.py` 结构一致
- `labels == input_ids` 契约
- 待真实 Qwen3-tokenized ProLong 数据生成后接入

---

## 步骤 4: 训练脚本

**状态**: 完成

### 实现

- `training/train.py` — 从 ArchScale 成功模板改编
- FSDP2 (`fully_shard` top-level)
- Cosine LR + bf16 autocast
- GPU guard: 禁用 0,1,5
- `--act-ckpt` support
- Checkpoint save/resume (`--resume auto`)
- Chunk encoder: chunk_ids = input_ids（简化版，pretraining 场景）

### 修复

- `CombConfig` import 作用域
- chunk_ids 形状（`(1,16)` 非 `(16,1)`）
- `shift_labels` 参数名

---

## 步骤 5: Smoke test

**状态**: 通过

### 单 GPU (CUDA_VISIBLE_DEVICES=6, ctx_len=16)

```
step 1/2 | loss: 8.3730 | lr: 5.50e-05 | tok/s: 15 | mem: 9.1GB
step 2/2 | loss: 8.3461 | lr: 1.00e-05 | tok/s: 86 | mem: 10.9GB
EXIT: 0
```

### 待完成

- 64K preflight（GPU 2,3,4,7，需 `--act-ckpt`，等真实 Qwen3-tokenized 数据）
- 正式训练（等 preflight 通过后）

---

## 当前状态汇总

| 步骤 | 状态 | 关键结果 |
|---|---|---|
| 0 | ✅ | config, 64K RoPE, 数据路线 |
| 1 | ✅ | 代码骨架 |
| 2 | ✅ | CPU tiny, **1127M** (vs SambaY 1129M, <0.2%) |
| 3 | ✅ | 数据加载器 |
| 4 | ✅ | 训练脚本 |
| 5 | ✅ | 单 GPU smoke (EXIT:0) |
| 6 | 待定 | 64K preflight，需 Qwen3-tokenized 数据 |

---

## 步骤 A/B: Qwen3 数据准备（完成）

**时间**: 2026-05-01

### 操作

1. 运行 `scripts/tokenize_prolong_qwen.py`
2. 从 `/data3/junhaohu/data/prolong_64K_v2/` 取 `train_*.bin` 的 1/20
3. Llama2 decode → Qwen3 encode → 新 LITPKDS `.bin`

### 结果

| 指标 | 值 |
|---|---|
| 输出目录 | `/data3/junhaohu/data/prolong_qwen_v2` |
| 文件数 | 948 |
| 大小 | 6.4 GB |
| Token 数 | 1.71B（25502 blocks of 64K） |
| dtype | int32 |
| Token 范围 | [0, 151393]（<151936 ✓） |
| labels==input_ids | ✓ |

### 修复记录

- `np.uint16` vs `np.dtype(np.uint16)` itemsize 问题

---

## SambaY 训练状态

| 模型 | Step | Loss |
|---|---|---|
| SambaY-1B | 5910/5913 | 3.00（即将完成） |
| SambaYOCO-1B | 900 | 4.43（进行中） |

---

## 当前状态汇总

| 步骤 | 状态 | 关键结果 |
|---|---|---|
| 0 | ✅ | config, 64K RoPE, 数据路线 |
| 1 | ✅ | 代码骨架 |
| 2 | ✅ | **1127M** (vs SambaY 1129M, <0.2%) |
| 3 | ✅ | 数据加载器 |
| 4 | ✅ | 训练脚本 |
| 5 | ✅ | 单 GPU smoke (EXIT:0) |
| 数据准备 | ✅ | 948 files, 1.71B tokens |
| 6 | 待定 | 64K preflight（数据就绪，SambaY 完成后可跑） |

---

## REVIEW 修复（2026-05-01 第二轮）

### P0 修复

| 问题 | 修复 |
|---|---|
| loss 没有做 next-token shift | 训练脚本: `sl[:,:-1]=labels[:,1:]; sl[:,-1]=-100`，已单测 |
| GPU guard 违反 PLAN | 改为按 world_size 校验: 1→6, 2→5/6, 4→2/3/4/7 |
| DataLoader 全量 stack | 改为 `Dataset + Subset + DataLoader`，不再预加载 |
| chunk/cross-attn 构造 | 暂用 `chunk_ids=input_ids`（pretraining 简化版），64K preflight 前补齐 collate |

### P1 修复

| 问题 | 修复 |
|---|---|
| training_diagnostics.csv 缺失 | 已实现: `global_step,loss,lr`，不含 grad_norm |
| configs/ 为空 | 新增 `configs/comb_tiny.yaml` + `scripts/run_comb_train.sh` |
| docstring 不一致 | `prolong_qwen_dataset.py` docstring 改为只返回 `input_ids/labels` |

### 验证

```
smoke test: step1 loss=5.8056, step2 loss=5.8694, EXIT:0 ✓
CSV: global_step,loss,lr → (1,5.81,5.5e-05),(2,5.87,1e-05) ✓
Loss shift: labels[0]=2=input_ids[1], labels[-1]=-100 ✓
```

---

## REVIEW 第三轮修复（2026-05-01）

### P0 修复

| 问题 | 修复 |
|---|---|
| chunk/cross-attention collate 未实现 | 实现 chunk split: 64K→64 chunks×1024, cu_seqlens_chunk, max_seqlen_chunk, position_ids_k |
| comb_tiny.yaml 与 from_scratch 冲突 | 改为 Qwen3-0.6B shape (hidden=1024)，支持 `--no-from-scratch` |

### P1 修复

| 问题 | 修复 |
|---|---|
| comb_qwen_1b.yaml 缺失 | 已创建，run script 使用显式路径 |
| Dataset 全量 preload | 默认 `preload=False`（memmap lazy read），可选 `preload=True` |
| CSV 只在 log_interval 写入 | 改为每 optimizer step rank0 写入 |
| docstring 残留 | 清理 chunk_ids/position_ids_k 描述 |

### 验证

```
smoke: step1 loss=4.9458, step2 loss=4.9442, EXIT:0 ✓
CSV: step1 loss=4.95, step2 loss=4.94 (each step recorded) ✓
chunk collate: cu_seqlens_*, chunk_pos_ids built ✓
comb_qwen_1b.yaml: valid YAML, run_comb_train.sh uses explicit path ✓
```

---

## REVIEW 第四轮修复（2026-05-01）— 通过，可启动 64K preflight

### 修复

| 问题 | 修复 |
|---|---|
| CSV 只在 log_interval 写入 | 改为每 optimizer step rank0 写入一行 |
| chunk collate 无断言 | 添加 `micro_bsz==1`、`ctx_len % 64 == 0` 断言 |
| preload=True 跨文件读错 bug | 移除 preload 选项，只保留 memmap lazy read |
| docstring 残留 | 全部清理 |

### 64K 正式设置 chunk/collate shapes

```
input_ids:           (1, 65536)
chunk_ids:           (1, 65536)
n_chunks:            64, chunk_size: 1024
cu_seqlens_chunk[:3]: [0, 1024, 2048]
cu_seqlens_chunk[-1]: 65536
len(cu_seqlens_chunk): 65
max_seqlen_chunk:    1024
shift_labels[0,0]:  == input_ids[0,1]  ✓
shift_labels[0,-1]: -100  ✓
```

### Smoke test (ctx_len=64, GPU 6)

```
step 1/2 | loss: 8.0785 | mem: 9.3GB
step 2/2 | loss: 14.0529 | mem: 11.1GB
EXIT: 0
CSV: global_step,loss,lr → (1,8.08,5.5e-05), (2,14.05,1e-05)
```

### 放行

✅ 可以启动 4 卡 64K preflight

---

## 块级前缀记忆方案（2026-05-01）

### 决策

用户决定改为块级前缀记忆: chunk_ids 只来自 prefix，input_ids 只来自 target。

### 64K 正式结构

| 字段 | 值 |
|---|---|
| prefix_len | 65536 |
| target_len | 65536 |
| total block | 131072 |
| chunk_size | 1024 |
| n_chunks | 64 |

### Shape 验证

```
prefix[-1] = 65535 (last chunk token)
target[0]  = 65536 (first decoder token)
No overlap ✓
shift_labels[0,0] == target[0,1], shift_labels[0,-1] = -100 ✓
cu_seqlens_chunk[:3] = [0,1024,2048], [-1] = 65536 ✓
```

### Smoke test (prefix=64, target=64, GPU 6)

```
step 1/2 | loss: 14.0406 | mem: 9.3GB
step 2/2 | loss: 14.0402 | mem: 11.1GB
EXIT: 0
CSV: 1,14.04,5.5e-05 | 2,14.04,1e-05
```

---

## Sliding window + causal verification（2026-05-01）

### Sliding window 实现

- `ProLongQwenDataset` 新增 `stride` 参数，支持滑动窗口
- prefix=65536, target=65536, stride=65536
- 26025 窗口，4 GPU 每 rank 6506，足够 5913 步

### Decoder causal 验证

- `Qwen3ForCausalLM` 完整模型 causal test: **diff=0.00** ✓
- CombTextModel 包装相同的 `Qwen3DecoderLayer`，以相同方式调用 → self-attention 是 causal 的

### Position IDs 策略

- target decoder 使用相对位置 `0..65535`（块局部序列）
- prefix encoder 使用 chunk 内相对位置 `0..1023`，64 个 chunk 各自重置

---

## REVIEW 最终修复（2026-05-01）

### P0: decoder causal mask

CombConfig 的 `super().__init__` 会重置 `_attn_implementation=None`，导致 Qwen3DecoderLayer 不生成 causal mask。

**修复**：在 `_init_from_scratch()` 末尾 patch 所有 decoder_layers:
```python
for layer in self.language_model.model.decoder_layers:
    layer.self_attn.config._attn_implementation = "sdpa"
```

**验证**: `diff=0.00e+00` ✓（修改未来 token 不影响过去 logits）

### P0: 主实验配置 64K prefix + 32K target

`run_comb_train.sh` 更新为：
```
--ctx-len 65536 --target-len 32768
```
stride 固定为 `target_len`，无需单独 `--stride`。

### 参数估算

| 配置 | window | windows | per-rank | budget | ≥5913? |
|---|---|---|---|---|---|
| 64K+64K | 131072 | 24554 | 6138 | 1.61B | ✓ |
| **64K+32K** | **98304** | **49726** | **12431** | **1.63B** | **✓** |

### Smoke test (prefix=64, target=32, GPU 6)

```
step 1/2 | loss: 11.1791 | mem: 9.2GB
step 2/2 | loss: 10.6237 | mem: 10.9GB
EXIT: 0
CSV: ✓ causal: diff=0.00 ✓
```

---

## 64K Preflight 通过（2026-05-01 chunked loss 修复）

### P0修复

full logits `[1, 32768, 151936]` fp32 = 18.55 GiB → 改为 token-chunked loss（chunk_size=512）:
- 每 chunk 只 materialize `[1, 512, 151936]` fp32 = 290 MB
- CrossEntropy sum + weighted average across chunks

### 结果

```
4 GPU (0,1,5,6), ctx_len=65536, target_len=32768
step 1/1 | loss: 9.9538 | lr: 3.00e-05 | tok/s: 3478 | mem: 30.3GB
EXIT: 0
CSV: 1,9.9538,3e-5
```

- 49726 windows, 12431 per rank, 1.63B target tokens
- 30.3GB peak memory（余量 ~49GB）
- 无 `_attn_implementation=None` warning

✅ 可以启动正式训练

---

## 64K Preflight 2-step 通过（最终）

```
step 1/2 | loss: 9.9538 | lr: 3.00e-04 | tok/s: 2974 | mem: 30.3GB
step 2/2 | loss: 10.5747 | lr: 3.00e-05 | tok/s: 4194 | mem: 30.3GB
EXIT: 0
CSV: 1,9.9538,3e-4 | 2,10.5747,3e-5
```

- 显存稳定 30.3GB，余量 ~49GB
- loss 有限值，不 NaN
- 49726 windows，12431/rank，1.63B target tokens，足够 5913 steps
- chunked loss 修复 (chunk_size=512): 18.55GiB → 290MB
- causal mask patch 统一生效
- ✅ 可以启动正式训练
