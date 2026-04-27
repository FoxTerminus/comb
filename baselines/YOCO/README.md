# YOCO Baseline

## 1. YOCO 的架构

这个目录实现的是一个纯 YOCO-style baseline，不依赖 `CombLlama` 的
chunk encoder 路径。

当前版本的结构固定为：

- 底座初始化来源：`Llama-3.1-8B-Instruct`
- 总层数：`32`
- self-decoder：前 `16` 层
- cross-decoder：后 `16` 层

核心思想是：

- self-decoder 先对输入 token 做局部建模
- self-decoder 使用 sliding-window attention（`SWA`）
- self-decoder 负责持有可复用的历史 memory / cache
- cross-decoder 不再走一套新的 full-history self-attention，而是读取 self-decoder 产生的 memory

因此，这个 baseline 的重点不是 chunk 压缩，而是验证一种 YOCO 式的
decoder-decoder 结构在当前仓库里是否正确、可训练、可推理。

和 `CombLlama` 的本质区别是：

- `CombLlama` 依赖 `chunk_ids -> chunk_model -> cross_attention_states`
- `YOCO` 完全不走这条路径
- `YOCO` 只吃普通 decoder-only 输入
- `YOCO` 的 cross-decoder memory 来自 self-decoder 本身，而不是外部 chunk encoder

## 2. YOCO 的功能

当前这个 baseline 已经具备以下能力：

- 从 config 直接实例化 YOCO 模型
- 普通前向训练，输出 logits 和 loss
- `use_cache=True` 的缓存推理
- HuggingFace 风格的 `generate()`
- 从 Llama checkpoint 初始化为 YOCO checkpoint
- 单卡训练
- Tensor Parallel 训练/推理适配

主要文件：

- [models/YOCO.py](/data3/junhaohu/comb/baselines/YOCO/models/YOCO.py)：核心模型与 cache / generation 逻辑
- [models/YOCO_megatron.py](/data3/junhaohu/comb/baselines/YOCO/models/YOCO_megatron.py)：TP 适配
- [scripts/init_yoco_from_llama.py](/data3/junhaohu/comb/baselines/YOCO/scripts/init_yoco_from_llama.py)：Llama -> YOCO 初始化
- [training/data.py](/data3/junhaohu/comb/baselines/YOCO/training/data.py)：YOCO 的 collate 逻辑
- [training/train_yoco_megatron.py](/data3/junhaohu/comb/baselines/YOCO/training/train_yoco_megatron.py)：训练脚本

## 3. 如何训练 YOCO

### 3.1 先从 Llama 初始化 YOCO

先把 Llama checkpoint 映射成 YOCO checkpoint：

```bash
python ./baselines/YOCO/scripts/init_yoco_from_llama.py \
  --llama-path meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /path/to/YOCO-Llama-8B-Init \
  --dtype bfloat16
```

这个脚本会：

- 构建 YOCO 模型
- 从 Llama 拷贝 embedding / norm / lm_head
- 拷贝 self-decoder 前 16 层
- 用 Llama 后 16 层初始化 cross-decoder 的 norm / mlp / cross-attn 投影
- 保存新的 YOCO checkpoint

### 3.2 YOCO 的训练输入

YOCO 训练使用 `baselines/YOCO/data` 下的专属预处理，不再复用
`comb/data` 的 `chunk_ids` / chunk-history 格式。

默认数据语义是：

- 使用同一批训练数据源
- 重新用 chat template 构造完整 decoder-only `input_ids`
- system/user token 全部 mask 为 `-100`
- assistant token 预处理成 causal next-token `shift_labels`
- 不产生 `chunk_ids`

预处理后的缓存路径是：

```text
$HF_HOME/datasets/yoco_<dataset_name>_<model_name>
```

如果需要重建缓存，可以在训练命令里加 `--force-reprocess-data`。

训练 batch 主要字段是：

- `input_ids`
- `shift_labels`
- `position_ids`
- `cu_seqlens_q`
- `max_seqlen_q`

其中实际送进模型的 `input_ids` 已经包含了前拼的 chunk 历史。对应的 collate
在 [training/data.py](/data3/junhaohu/comb/baselines/YOCO/training/data.py)。

### 3.3 跑训练

训练入口：

[train_yoco_megatron.py](/data3/junhaohu/comb/baselines/YOCO/training/train_yoco_megatron.py)

最小 synthetic smoke / overfit：

```bash
PYTHONPATH=/data3/junhaohu/comb \
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  ./baselines/YOCO/training/train_yoco_megatron.py \
  --tp-size 1 \
  --model-name /path/to/llama-or-tiny-llama \
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

用已经初始化好的 YOCO checkpoint 做正式训练：

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

注意：YOCO-native 数据集已经直接输出 next-token labels，正式训练保持默认
`--label-shift-mode existing` 即可，不要再传 `--label-shift-mode next-token`，
否则会二次右移。

如果本地 checkpoint 是旧版 YOCO 结构生成的，需要先用当前
`init_yoco_from_llama.py` 重新初始化一份 checkpoint。新版结构按官方源码将
cross-decoder 的 `kv_layer_norm/k_proj/v_proj` 放在 cross-decoder 级别，并被
所有 cross-attention layers 共享；self-decoder 的 SWA K/V 使用 full attention
heads。旧版 checkpoint 中每层独立的 cross K/V 不会再被使用。

训练原则：

- YOCO 走 full-parameter training
- 不冻结参数
- 不再单独编码 `chunk_ids`
- 但会把 `chunk_ids` 全量并入 decoder 上下文
- 先做单卡 correctness，再做 TP 和大规模训练

## 4. 如何把 YOCO 作为 CombLlama 的 baseline 进行比较

比较时，重点是让 YOCO 和 `CombLlama` 的差异只落在架构上，而不是落在初始化、数据或训练策略上。

### 4.1 比较对象

至少保留三条线：

- 原始 Llama
- YOCO
- CombLlama

其中：

- Llama 是 plain decoder-only 参考
- YOCO 是 pure decoder-decoder baseline
- CombLlama 是当前仓库已有的 chunk-based 方案

### 4.2 保持可比性的原则

YOCO 和 CombLlama 对比时，尽量保持下面这些项一致：

- 相同或尽量接近的初始化来源
- 相同训练数据范围
- 相同优化器类型
- 相同学习率 schedule
- 相同 global batch size / micro batch size
- 相同训练步数
- 相同精度设置，例如 `bf16`

只有输入实现允许不同：

- YOCO：把 chunk 历史并入 decoder 上下文的 decoder-only path
- CombLlama：chunk + decoder path

### 4.3 训练阶段怎么比

先比训练可行性，再比效果：

1. 小规模 smoke / overfit
2. 相同步数下的训练 loss 曲线
3. validation loss 或 perplexity
4. checkpoint resume 稳定性

最基本的训练对比表建议记录：

| Model | Init | Data Path | Params | Train Loss | Val Loss | Notes |
|---|---|---|---:|---:|---:|---|
| Llama | Llama | decoder-only | ... | ... | ... | reference |
| YOCO | Llama | decoder-only | ... | ... | ... | pure YOCO |
| CombLlama | Llama-compatible | chunk + decoder | ... | ... | ... | current model |

### 4.4 推理阶段怎么比

推理比较建议关注：

- prefill latency
- decode latency
- peak GPU memory
- 长 prompt 下的 cache 行为
- generation 是否稳定

这样能回答两个问题：

- YOCO 相比 Llama，YOCO-style memory 路径是否有价值
- YOCO 相比 CombLlama，是否在不依赖 chunk encoder 的前提下提供更简单或更稳健的 baseline

### 4.5 最终结论该怎么下

如果要把 YOCO 作为正式 baseline 保留，至少要回答：

- YOCO 是否训练稳定
- YOCO 是否能达到可接受的质量
- YOCO 在推理速度或显存上是否比 plain Llama 更有优势
- YOCO 相比 CombLlama 的 tradeoff 是什么

一句话概括：

YOCO 在这里的角色，不是替代 Comb 的 chunk 设计，而是提供一个干净、独立、可复现的 decoder-decoder baseline，用来和 `CombLlama` 做结构层面的直接比较。
