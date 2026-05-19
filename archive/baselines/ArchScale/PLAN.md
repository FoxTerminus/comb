# SambaY vs Samba+YOCO 对比实验执行计划（REVIEW 修订版）

> 修订时间：2026-04-29 | 修订依据：`REVIEW.md`

## 一、背景与目标

### 实验目的
在 d=16（~1B 参数）量级下，使用 ProLong-64k 数据集从头训练两个模型架构并对比：
- **SambaY**：Mamba self-decoder + YOCO cross-decoder + GMU 门控 + NoPE
- **Samba+YOCO**（即 `sambayoco`）：Mamba self-decoder + YOCO cross-decoder + 无 GMU

实验性质：**参数量匹配的 GMU ablation**。两个模型的 `ar` 参数略有不同（124 vs 126），这是 ArchScale 源码中为匹配参数量做的原始设计，会导致 `n_embd`（1984 vs 2016）和 `intermediate_size`（7936 vs 8064）存在差异。结论表达时需将这一点作为已知 confound 进行讨论，不能声称"严格唯一变量"。

### 硬件约束
- GPU：**仅使用 2, 3, 4, 7（共 4 张 A100-80GB）**，绝不动用 0, 1, 5, 6
- CUDA 12.8, PyTorch 2.6.0+cu124
- 已有 `flashattn` conda 环境（含 Flash Attention 2.7.4）

### Tokenizer 约定
- Tokenizer：**Llama-2-7B-HF**（`meta-llama/Llama-2-7b-hf`）
- `vocab_size=32000`（与 ProLong_64K_v2_Llama2_Tokenizer 数据一致）
- 特殊 token：pad_token_id=0（Llama-2 无原生 pad token，设为 0）
- 训练脚本需在初始化时校验 `tokenizer.vocab_size == model.config.vocab_size`

### 重要约束
- **不得使用 `baselines/SambaY/` 和 `baselines/YOCO/` 中的任何代码**（均为废案）
- 模型代码全部从 ArchScale GitHub 源码（`microsoft/ArchScale`）移植重写

---

## 二、技术路线

1. **模型代码**：从 `microsoft/ArchScale` 的 `lit_gpt/` 目录移植核心模块
2. **训练代码**：全新手写，使用 PyTorch FSDP2 (`fully_shard`) + Cosine LR
3. **数据管线**：新写 ProLong-64k 专用 packed dataset

---

## 三、文件规划

所有文件均为新建，放在 `baselines/ArchScale/` 下：

| 文件路径 | 用途 |
|---|---|
| `models/__init__.py` | 模型模块入口 |
| `models/config.py` | Config 类（sambay_d16, sambayoco_d16） |
| `models/mamba.py` | Mamba-1 层 |
| `models/gmu.py` | GMU + GMUWrapper |
| `models/attention.py` | CausalSelfAttention + yoco_cross |
| `models/model.py` | GPT + Block（YOCO split） |
| `data/__init__.py` | 数据模块入口 |
| `data/prolong_dataset.py` | ProLong-64k packed dataset |
| `training/__init__.py` | 训练模块入口 |
| `training/train.py` | 分布式训练主脚本 |
| `configs/sambay_1b.yaml` | SambaY 1B 模型配置 |
| `configs/sambayoco_1b.yaml` | Samba+YOCO 1B 模型配置 |
| `scripts/download_prolong.sh` | ProLong 数据下载与子采样 |
| `scripts/run_sambay_train.sh` | SambaY 训练启动脚本 |
| `scripts/run_sambayoco_train.sh` | Samba+YOCO 训练启动脚本 |
| `eval/eval_comparison.sh` | 对比评估脚本 |
| `tools/inspect_model.py` | 模型结构验证工具（打印每层 mixer 类型） |

---

## 四、实施步骤

### 步骤 1：环境准备

**操作内容：**
1. 安装 `mamba-ssm==2.3.1` 和 `causal-conv1d==1.6.1` 到 `flashattn` conda 环境
2. 验证安装：`from mamba_ssm.ops.selective_scan_interface import selective_scan_fn`；`from causal_conv1d import causal_conv1d_fn`
3. 创建目录结构

### 步骤 2：模型配置定义

**关键参数对照（修订：block_size 统一为 65536）：**

| 参数 | SambaY (sambay_d16) | Samba+YOCO (sambayoco_d16) |
|---|---|---|
| block_size | **65536** | **65536** |
| n_embd | 1984 (ar=124 × 16) | 2016 (ar=126 × 16) |
| n_layer | 16 (8 self + 8 cross) | 16 (8 self + 8 cross) |
| n_head | 16 | 16 |
| head_dim | 128 | 128 |
| n_query_groups | 4 (GQA) | 4 (GQA) |
| intermediate_size | 7936 (mlp_expand=4) | 8064 |
| rnn_per_layer | 2 | 2 |
| rnn_type | mamba | mamba |
| yoco | True | True |
| gmu_yoco | **True** | **False** |
| local_window | 128 | 128 |
| nope | True | True |
| vocab_size | 32000 | 32000 |
| 参数量 | ~973M | ~973M |

> **block_size 说明**：ArchScale 原始 `sambay_d16` 的 `block_size=4096`。本实验目标为 64K 训练，因此将 `block_size` 改为 65536。需在实现时确保所有 attention mask、position 计算、RoPE 频率不依赖 4K 常量。Self-decoder Mamba 层不依赖 block_size（SSM 无固定长度限制）。

### 步骤 3：模型实现

同之前，从 ArchScale 移植 6 个模块。

**程序化层验证要求**：实现 `tools/inspect_model.py`，打印每层的 `layer_idx`、`position`（self/cross）、`token_mixer` 类型（Mamba/LocalAttn/FullAttn/GMU/CrossAttn）、`gmu_save`、`yoco_kv`、`yoco_cross`，与预期表逐项断言。

### 步骤 4：ProLong-64k 数据加载器

**LITPKDS 数据验证要求**：
- 读取前先验证 header：magic="LITPKDS"、version=1、dtype 合法
- 打印首尾各 10 个 token、总 token 数、文件数
- 检验 `labels = input_ids[1:]` 的 shift 不跨文件边界（每个 .bin 文件末尾不 shift 到下一个文件）
- 不同 rank 的无重复采样验证（DeterministicSampler）
- 同一 seed 的可复现性验证

### 步骤 5：训练脚本编写

**训练预算（修订）：**

| 参数 | 值 | 计算过程 |
|---|---|---|
| 序列长度 | 65536 | ProLong 固定 |
| 总训练 tokens | 1.55B | ProLong-64k 的 1/20 |
| Per-GPU micro batch | 1 | 64K 显存约束 |
| GPU 数 | 4 | |
| 每 micro-step tokens | 262,144 | 4 × 1 × 65536 |
| Gradient accumulation | **1** | 每个 micro-step 就是一次 optimizer step |
| Effective batch size | 262,144 tokens/step | |
| Optimizer steps | **5,913** | 1.55e9 / 262144 |
| Warmup steps | **591** (10%) | 5913 × 0.1 |
| Peak LR | 3e-4 | |
| Min LR | 3e-5 | peak_lr × 0.1 |
| LR schedule | Cosine decay | |
| Optimizer | AdamW (β1=0.9, β2=0.95) | |
| Weight decay | 0.1 | |
| Gradient clip | 1.0 | |
| Precision | bfloat16 (autocast) | |

**Checkpoint 要求**：
- 保存内容：model_state_dict、optimizer_state_dict、scheduler_state_dict、global_step、micro_step、rng_states（torch、numpy、python random）、config snapshot
- 保存时机：每 500 optimizer steps + 训练结束时
- Resume 逻辑：自动检测最新 checkpoint，恢复所有状态后从断点继续

**GPU 保护**：
- 启动时校验 `CUDA_VISIBLE_DEVICES` 不含物理 GPU 0,1,5,6
- 如果检测到禁用 GPU，立即 `raise RuntimeError` 并打印错误信息

### 步骤 6：Smoke Test 与 Preflight

**Smoke test（单 GPU）：**
1. tiny config（d=4, hidden_size=128, ctx_len=64）：前向 + 反向 + 10 步训练
2. 真实 1B config + ctx_len=1024：单次前向 + 反向不 OOM

**64K preflight（4 GPU，在正式训练前必须完成）：**
- 用真实 1B config + ctx_len=65536，4 GPU 跑 ≥2 optimizer steps
- 记录每 GPU 峰值显存、tokens/s、loss 值
- 验证 gradient checkpointing + FlashAttention 正常工作
- 验证 KV cache shape 和 gmu_mems shape 正确
- 若 OOM：尝试 activation checkpointing、减少 local_window、降低到 32K 等 fallback

**其他验证项：**
- 参数量脚本：分别输出 SambaY 和 Samba+YOCO 的 total/trainable params
- 形状测试：batch=2, seq=64 验证 logits、loss、kv_cache、gmu_mems shapes
- 因果性测试：修改位置 t+1 的 input token，位置 t 的 logits 不应变化
- GMU ablation 测试：`gmu_yoco=True/False` 仅影响 cross-decoder 的偶数层（10,12,14）
- 数据确定性测试：相同 seed 和 world size 下数据加载可复现
- GPU guard 测试：设置 `CUDA_VISIBLE_DEVICES=0,2,3` 必须 fail fast

### 步骤 7：启动正式训练

```bash
export CUDA_VISIBLE_DEVICES=2,3,4,7
torchrun --nnodes=1 --nproc_per_node=4 \
  baselines/ArchScale/training/train.py \
  --config baselines/ArchScale/configs/sambay_1b.yaml \
  --data-dir /data3/junhaohu/data/prolong_64K_v2_subset \
  --output-dir /data3/junhaohu/model/SambaY-1B-Prolong
```

Samba+YOCO 使用相同命令，仅替换 `--config` 为 `sambayoco_1b.yaml`。

### 步骤 8：评估对比

- Loss 曲线对比
- RULER, SCBench, LongBench 评测

---

## 五、风险与缓解

| 风险 | 概率 | 缓解措施 |
|---|---|---|
| ArchScale 模型代码移植有遗漏 | 中 | `inspect_model.py` 逐层验证；分模块单元测试 |
| 64K 序列 FSDP2 OOM | **中高** | activation checkpointing + FlashAttention；64K preflight 提前检测；fallback 到 32K 或减小 local_window |
| mamba_ssm 安装失败 | 低 | PyTorch 2.6+cu124 与 mamba_ssm 2.3.1 兼容 |
| 训练 loss 不收敛 | 中 | 先 tiny config smoke test 排查 |
| 1.55B tokens 不够充分训练 | 中 | 对架构对比目的足够，报告中说明此限制 |

---

## 六、验证清单

- [ ] mamba_ssm + causal_conv1d 安装成功
- [ ] 模型每个模块可独立 import 和实例化
- [ ] `inspect_model.py` 输出与计划层次表一致
- [ ] 参数量：SambaY ~973M，Samba+YOCO ~973M
- [ ] tiny config smoke test 通过
- [ ] 因果性测试通过
- [ ] GMU ablation 模块拓扑测试通过
- [ ] 数据确定性测试通过
- [ ] GPU guard fail-fast 测试通过
- [ ] LITPKDS header 解析正确
- [ ] 64K preflight 通过（≥2 optimizer steps，记录峰值显存和 tokens/s）
- [ ] 正式训练 loss 正常下降
- [ ] checkpoint 可保存和恢复
- [ ] 两个模型 loss 曲线和 benchmark 可对比

---

## 七、时间估算

| 阶段 | 预估时间 |
|---|---|
| 环境准备 + 依赖安装 | 1 小时 |
| 模型配置定义 | 0.5 小时 |
| 模型实现 + inspect 工具 | 4-6 小时 |
| ProLong 数据下载 + 数据加载器 | 2-3 小时 |
| 训练脚本编写 | 2-3 小时 |
| Smoke test + 64K preflight | 2 小时 |
| SambaY 训练 | 1.5-3 天 |
| Samba+YOCO 训练 | 1.5-3 天 |
| 评估 + 绘图 | 2 小时 |
| **总计** | **约 5-8 天** |
