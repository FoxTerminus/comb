# REPORT.md — 步骤 1-3 完成报告（第三轮复审修订版）

**时间**: 2026-04-30  
**环境**: `samba` (conda, Python 3.12)  
**GPU**: 4×A100-SXM4-80GB

---

## 步骤 1：环境准备

### 操作内容

1. 放弃 `flashattn`（Python 3.13，mamba-ssm ABI 不兼容），创全新 conda 环境 `samba`（Python 3.12）
2. 安装 PyTorch 2.6.0+cu124（GPU 后端 CUDA 12.4）
3. `causal-conv1d==1.6.1`：源码编译，CUDA 12.4，`-D_GLIBCXX_USE_CXX11_ABI=0`
4. `mamba-ssm==2.3.1`：GitHub 预编译 wheel，`cxx11abiFALSE`，免编译
5. `flash-attn==2.8.3`：源码编译，仅 sm_80/sm_90，`-D_GLIBCXX_USE_CXX11_ABI=0`
6. 附加依赖：einops 0.8.2, triton 3.2.0, transformers 5.7.0, pyyaml 6.0.3, ninja 1.13.0

### 验证结果

| 依赖 | 版本 | GPU 功能测试 |
|---|---|---|
| torch | 2.6.0+cu124 | 4×A100 可见, CUDA 12.4 |
| causal_conv1d | 1.6.1 | `causal_conv1d_fn` 前向/反向 ✅ |
| mamba_ssm | 2.3.1 | `selective_scan_fn` 前向/反向 ✅ |
| flash_attn | 2.8.3 | `flash_attn_func` 前向/反向 ✅ |

---

## 步骤 2：模型配置定义

### 操作内容

1. 实现 `models/config.py`：`Config` 类 + `from_name` + `from_yaml` + `estimated_params`
2. 创建 `configs/sambay_1b.yaml` 和 `configs/sambayoco_1b.yaml`

### 关键参数

| 参数 | SambaY | SambaYOCO | 说明 |
|---|---|---|---|
| block_size | 65536 | 65536 | 64K 训练 |
| vocab_size | 32000 | 32000 | Llama-2 tokenizer |
| n_layer | 16 | 16 | 8 self + 8 cross |
| n_head | 16 | 16 | |
| n_query_groups | 4 | 4 | GQA |
| head_dim | 128 | 128 | |
| ar | 124 | 126 | `n_embd = ar * n_layer` |
| n_embd | 1984 | 2016 | |
| intermediate_size | 7936 | 8064 | `mlp_expand=4` |
| rnn_per_layer | 2 | 2 | |
| rnn_type | mamba | mamba | |
| yoco | True | True | |
| gmu_yoco | **True** | **False** | 实验变量（参数量匹配 GMU ablation） |
| nope | True | True | |
| local_window | 128 | 128 | |

> **参数量说明**：ArchScale 的 ~973M 为非 embedding 参数量（μP++ 公式）。
> 本实现含 embedding/lm_head 总参数量 ~1.13B。两模型差 9.4M 来自 `ar`
> 差异（124 vs 126）。

---

## 步骤 3：模型实现

### 完成的模块

| 文件 | 职能 |
|---|---|
| `models/config.py` | Config 类 + `from_name`/`from_yaml` |
| `models/mamba.py` | Mamba-1 selective scan，`gmu_save` 保存 SSM 输出 |
| `models/gmu.py` | GMU + GMUWrapper |
| `models/attention.py` | Self-Attn + Cross-Attn, GQA, causal, local sliding window (FlashAttention-2) |
| `models/model.py` | Block（YOCO split）+ GPT + SwiGLUMLP + RMSNorm |
| `tools/inspect_model.py` | 程序化层验证工具 |

### 修复记录（三轮复审）

**第一轮**：cross-attn 加 causal mask、sliding window 实现、移除 `.bfloat16()` 硬编码、新增 `inspect_model.py`

**第二轮**：`inspect_model.py` KV producer 计数修正、sliding window 改为 FlashAttention-2 `window_size`（避免 64K dense mask OOM）、cross-attn 加 `kv_cache` 断言

**第三轮**：FlashAttention 形状修正——`flash_attn_func` 期望 `(B, T, H, D)` 而非 SDPA 的 `(B, H, T, D)`

### 验证结果

**GPU 前向 + 反向**（bf16）：
```
SambaY loss=10.4859  SambaYOCO loss=10.6126
fwd+bwd: OK
```

**因果性测试**（GPU）：
```
修改 token 20, 位置 < 20 的 logits diff=0.00e+00  ✅
```

**Sliding window 语义测试**（flash_attn, local_window=2）：
```
修改 token 0 → token 1 diff=40.5 (window 内, 应变化 ✅)
修改 token 0 → token 4 diff=0.0  (window 外, 不应变 ✅)
修改 token 0 → token 0 diff=64.5 (self-attn, 应变化 ✅)
```

**层次拓扑验证**（`tools/inspect_model.py`）：

SambaY (`sambay_d16`):
```
Layer  Mixer                    Tags                     Self/Cross
0      Mamba                    RNN                      self_decoder
1      CausalSelfAttention      LOCAL_SWA                self_decoder
2      Mamba                    RNN                      self_decoder
3      CausalSelfAttention      LOCAL_SWA                self_decoder
4      Mamba                    RNN                      self_decoder
5      CausalSelfAttention      LOCAL_SWA                self_decoder
6      Mamba                    RNN                      self_decoder
7      CausalSelfAttention      LOCAL_SWA                self_decoder
8      Mamba                    RNN                      cross_decoder
9      CausalSelfAttention      KV_PRODUCER              cross_decoder
10     GMUWrapper               GMU | CROSS_ATTN         cross_decoder
11     CausalSelfAttention      CROSS_ATTN               cross_decoder
12     GMUWrapper               GMU | CROSS_ATTN         cross_decoder
13     CausalSelfAttention      CROSS_ATTN               cross_decoder
14     GMUWrapper               GMU | CROSS_ATTN         cross_decoder
15     CausalSelfAttention      CROSS_ATTN               cross_decoder

GMU layers: 3  KV producer: 1  ✅
```

SambaYOCO (`sambayoco_d16`):
```
Layer  Mixer                    Tags                     Self/Cross
...    (L0-L9 同 SambaY)
10     CausalSelfAttention      CROSS_ATTN               cross_decoder
11     CausalSelfAttention      CROSS_ATTN               cross_decoder
12     CausalSelfAttention      CROSS_ATTN               cross_decoder
13     CausalSelfAttention      CROSS_ATTN               cross_decoder
14     CausalSelfAttention      CROSS_ATTN               cross_decoder
15     CausalSelfAttention      CROSS_ATTN               cross_decoder

GMU layers: 0  KV producer: 1  ✅
```

**CPU 测试**（attention + GMU + config，不含 CUDA-only Mamba kernel）：全部通过 ✅

---

## 目录结构

```
baselines/ArchScale/
├── models/
│   ├── __init__.py
│   ├── config.py
│   ├── mamba.py
│   ├── gmu.py
│   ├── attention.py
│   └── model.py
├── configs/
│   ├── sambay_1b.yaml
│   └── sambayoco_1b.yaml
├── tools/
│   └── inspect_model.py
├── training/
├── data/
├── eval/
├── scripts/
│   └── build_mamba.sh
├── PLAN.md
├── REVIEW.md
├── REPORT.md
├── SambaY.pdf
└── You Only Cache Once.pdf
```

---

## 数据契约（步骤 4 前置约定）

`GPT.forward()` 内部已经对 `labels` 做 causal shift：
```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```
因此 **dataset 应返回未 shift 的 `labels = input_ids`**，不要再预先 shift。

---

## 下一步

- 步骤 4：ProLong-64k 数据加载器（可开始，需遵守数据契约）
- 步骤 5：训练脚本编写（模型已就绪，可在步骤 4 完成后开始）
- 步骤 6：Smoke test / 64K preflight

---

## 步骤 4：ProLong-64k 数据加载器

**时间**: 2026-04-30
**状态**: 完成（REVIEW 修订版）

### 完成的操作

1. 实现 `data/prolong_dataset.py`：
   - `_read_header()` — LITPKDS header 解析（magic/version/dtype/packed_chunk_size）
   - `_file_num_tokens()` — 从文件大小推导真实 token 数（`(filesize - HDR_SIZE) // dtype.itemsize`），含整除校验
   - `ProLongPackedDataset` — IterableDataset，预扫描建立全局 block 索引
   - 真实 token 数 = `(payload_bytes) / dtype.itemsize`（**不是** header 中的 `chunk_size`，那是每个 packed chunk 的 token 数）
   - 数据契约：`labels = input_ids`（模型内部做 causal shift）
2. 实现 `data/__init__.py` — 导出 `ProLongPackedDataset`

### REVIEW 修复

| 问题 | 修复 |
|---|---|
| `chunk_size` 误当文件总 token 数 | 改为 `os.path.getsize()` 推导，header `chunk_size` 仅作 packed chunk 语义 |
| 多 chunk 文件漏读 | payload bytes / itemsize 覆盖全部 token |
| payload 对齐无校验 | `_file_num_tokens()` 检查整除 |
| `total_blocks` 返回 shard 后值 | 新增 `rank_blocks`，`total_blocks` 返回 shard 前全局值 |
| `data/__init__.py` 不导出 | 已导出 `ProLongPackedDataset` |

### 验证结果

| 测试项 | 结果 |
|---|---|
| LITPKDS header 解析 | ✅ |
| 多 chunk synthetic 文件（chunk_size=8, 16 tokens, block=4）→ 4 blocks | ✅ |
| 16 个 token 全覆盖，无遗漏 | ✅ |
| 相同 seed 确定性 | ✅ |
| 分布式 rank 无重复 | ✅ |
| labels == input_ids（契约） | ✅ |

### 待完成

- 实际下载 ProLong 数据（`jsun/Prolong_64K_v2_Llama2_Tokenizer`，49.1 GB zip）
- 子采样 1/20 后放入 `/data3/junhaohu/data/prolong_64K_v2_subset`
- 对至少 1 个真实 `.bin` 文件打印 header、token 数、首尾 token id 范围验证

---

## 步骤 5：训练脚本（第八轮 REVISION — 步骤 5 通过）

**时间**: 2026-04-30
**状态**: 步骤 5 通过 synthetic smoke（1/2/4 GPU）

### 第八轮修复

| 问题 | 修复 |
|---|---|
| `total_tokens`/`total_blocks` AttributeError | 修复为访问 `_total_tokens`/`_total_blocks` |
| `total_tokens` 重复计数 | 移除 block 构建循环中重复的 `total_tokens += num_tokens` |

### 正式验证结果

| 测试 | 结果 |
|---|---|
| 单卡 2 steps + checkpoint | ✅ |
| 单卡 `--resume auto` | ✅ |
| 2-GPU 2 steps (无中途 checkpoint) | ✅ |
| 2-GPU 2 steps + 中途 checkpoint | ✅ |
| 2-GPU `--resume auto` | ✅ |
| 4-GPU 2 steps + checkpoint | ✅ |

### 已知风险

1. **preload 仅适合 subset**：当前 `ProLongPackedDataset(preload=True)` + `train.py` 内 `torch.stack` 将全部数据加载到 CPU 内存。当前 1/20 train-only subset（3.8GB）在本机可行，但完整 ProLong（84GB 解压，~45B tokens）不能整包 preload。正式长训如使用更大 subset 需 `preload=False` + mmap/lazy loading。

2. **梯度裁剪已移除**：`clip_grad_norm_()` + FSDP2 多 rank 交互导致 2-GPU/4-GPU 卡死。正式训练前需恢复 FSDP2 安全实现（如 `torch.nn.utils.clip_grad_norm_` + all-reduce total norm）或明确禁用策略及梯度爆炸风险。

3. **subset 已确认为 train-only**：当前 `/data3/junhaohu/data/prolong_64K_v2_subset` 仅含 948 个 `train_*.bin`，无 validation 文件。

---

## 步骤 6：真实数据校验 + 64K preflight（通过）

**时间**: 2026-05-01
**状态**: 全部通过，可以启动正式训练

### 6.1 真实数据校验

解压 `prolong_64K_v2.zip`（46GB → 84GB，21347 个 `.bin` 文件）后：

| 验证项 | 结果 |
|---|---|
| Header (magic/version/dtype/chunk_size) | ✓ LITPKDS, v1, uint16, 2098176 |
| Token 范围 < 32000 | ✓ [1, 30122] |
| `labels == input_ids` | ✓ |
| Determinism (相同 seed) | ✓ |
| Block 形状 = (65536,) | ✓ |

### 6.2 子采样 1/20（train-only）

用户决策：只使用 train split，丢弃 validation。
从 `train_*.bin` 取 NR%20==1 → 948 文件，3.8GB，~2.0B tokens。

### 6.3 64K preflight

```
CUDA_VISIBLE_DEVICES=3,4,6,7  torchrun --nproc_per_node=4
--config sambay_d16 --ctx-len 65536 --act-ckpt
--data-dir /data3/junhaohu/data/prolong_64K_v2_subset
```

| 指标 | Step 1 | Step 2 |
|---|---|---|
| Loss | 10.80 | 9.61 |
| Tokens/s | 3,698 | 12,840 |
| Peak memory | 43.9 GB | 46.1 GB |

4-GPU × 80GB 可行（内存余量 ~34GB）。Activation checkpointing 必需（无 `--act-ckpt` 时 OOM）。

---

## 下一步

- 启动 SambaY 正式训练（`--total-steps 5913`）
- 启动 Samba+YOCO 正式训练（完全相同超参数）

