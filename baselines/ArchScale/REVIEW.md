# REVIEW: 步骤 4 修复后复审

审核对象：`baselines/ArchScale/REPORT.md` 及 `data/prolong_dataset.py`  
审核时间：2026-04-30  
审核结论：**步骤 4 的代码实现通过 synthetic 复审，可以进入步骤 5 训练脚本开发；真实 ProLong 数据下载与真实 `.bin` 校验仍是正式训练/64K preflight 前的门禁。** 上轮指出的 `chunk_size` 语义问题已修复，现在数据集会从文件大小推导真实 token 数，不再把 header `chunk_size` 当成文件总 token 数。

## 我运行的验证

- 多 chunk synthetic LITPKDS 文件：header `packed_chunk_size=8`，payload 16 tokens，`block_size=4`。
- 当前 `ProLongPackedDataset` 正确推导 `file_tokens=16`、`total_blocks=4`、`rank_blocks=4`。
- 所有 16 个 token 都可被 block 覆盖；输出顺序会因 seed shuffle 改变，这是预期行为。
- `labels == input_ids` 契约成立，符合 `GPT.forward()` 内部 shift 设计。
- 双 rank synthetic 分片无 overlap，且相同 seed 下 deterministic。
- `data/__init__.py` 已导出 `ProLongPackedDataset`。

## 已修复/改善

- [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:61) 现在把 header 字段命名/解释为 `packed_chunk_size`。
- [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:83) 新增 `_file_num_tokens()`，通过 `(file_size - HDR_SIZE) // dtype.itemsize` 计算真实 payload token 数，并检查 payload byte 对齐。
- [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:139) block 索引改为基于真实 `num_tokens // block_size`。
- [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:186) `total_blocks` 现在表示 rank sharding 前的全局 block 数，并新增 `rank_blocks` 表示当前 rank block 数。

## 仍需注意

- 真实 ProLong 数据尚未完成下载，因此还不能确认真实文件的 header、dtype、payload token 数、token id 范围和 block 数。
- 当前 `summary()` 中 `total_tokens` 是 raw payload tokens；若正式训练只使用完整 `block_size` blocks，训练 token 预算应使用 `total_blocks * block_size`，不要误用包含尾部丢弃 token 的 raw count。
- 多 worker 切分采用连续分段；功能上可以，但后续如果要严格 epoch-level load balance，可以在训练脚本中记录每 rank/worker 的实际 sample 数。

## 真实数据下载监督

- 目标仓库：`jsun/Prolong_64K_v2_Llama2_Tokenizer`
- 远端文件：`prolong_64K_v2.zip`
- 远端大小：`49,134,434,378` bytes，约 46 GiB / 49.1 GB
- 下载目录：`/data3/junhaohu/data/prolong_64K_v2_raw`
- 日志/缓存：`/data3/junhaohu/data/prolong_64K_v2_raw/.cache/huggingface/`
- 当前状态：下载和解压已完成；zip 中 `.bin` 数与解压目录 `.bin` 数均为 `21347`。
- 注意：`wget` 和 `curl` 直接下载均快速退出并留下 0 byte 文件；已将 0 byte 文件改名为 `prolong_64K_v2.zip.zero`，避免被误认为下载完成。

下载完成后必须补做真实数据校验：
- 打印至少 1 个真实 `.bin` 的 header：magic、version、dtype、packed_chunk_size。
- 打印 payload token 数、`num_tokens // 65536` block 数、首尾 token id。
- 检查 token id 范围 `< vocab_size=32000`。
- 用真实数据实例化 `ProLongPackedDataset(block_size=65536)`，确认 `labels == input_ids`、rank 无 overlap、seed 可复现。

## 是否可以进入下一步

- **步骤 5 训练脚本：可以开始开发。** 可以基于当前 dataset 接口写 FSDP2、checkpoint、scheduler、GPU guard、logging 等训练框架。
- **真实 dataloader 联调：等待真实数据下载完成后再做。**
- **步骤 6 smoke/64K preflight：不能跳过真实数据校验。** 先用 synthetic/tiny 数据跑训练脚本 smoke，再用真实 ProLong subset 做正式 preflight。

---

# REVIEW: 步骤 5/6 真实数据与 64K preflight 复审

审核对象：`baselines/ArchScale/training/train.py`、`data/prolong_dataset.py`、真实 ProLong 解压数据、`REPORT.md` 步骤 5/6  
审核时间：2026-05-01  
审核结论：**步骤 6 的真实数据校验与 64K preflight 可以视为通过，可以开始正式训练。** 数据已完整下载并解压，subset 已建立，报告中的 4-GPU 64K preflight 有 checkpoint 产物支撑。用户已明确决策：**不保留 validation split，正式训练只使用 train split**；当前 subset 已为 train-only。训练诊断 CSV 已简化为 `global_step,loss,lr`，可用于后续 training loss 曲线对比。

## 我运行的验证

- Dataset 属性回归复测：`len(ds)=8`、`total_tokens=128`、`total_blocks=8`、`ds[0]["input_ids"].shape == [16]`、`labels == input_ids` 均通过。
- 2-GPU tiny synthetic，带中途 checkpoint：`CUDA_VISIBLE_DEVICES=3,4`、`nproc_per_node=2`、`save_interval=1` 成功完成 step 1/2 checkpoint 和 final checkpoint。
- 2-GPU `--resume auto`：从 step 2 恢复到 step 3 成功。
- 4-GPU tiny synthetic：`CUDA_VISIBLE_DEVICES=3,4,6,7`、`nproc_per_node=4`、`save_interval=1` 成功完成 step 1/2 checkpoint 和 final checkpoint。
- 完整解压校验：`prolong_64K_v2.zip` 中 `.bin` 数为 `21347`，解压目录 `/data3/junhaohu/data/prolong_64K_v2/prolong_64K_v2` 中 `.bin` 数也为 `21347`。
- subset 校验：`/data3/junhaohu/data/prolong_64K_v2_subset` 中 `.bin` 数为 `948`，大小约 `3.8GB`。
- subset 构成：`train_*.bin` 为 `948` 个，`validation_*.bin` 为 `0` 个，符合 train-only 决策。
- 真实 `.bin` 抽检：magic=`LITPKDS`、version=`1`、dtype=`uint16`、chunk=`2098176`、每文件可形成 `32` 个 64K blocks；抽样 token max 均 `< 32000`。
- preflight 产物：`/data3/junhaohu/tmp/preflight_ckpts/step_000002.pt` 存在，大小约 `3.2GB`，修改时间为 `2026-05-01 09:39 CST`，与 REPORT 中 2-step 64K preflight 对齐。
- 训练诊断 CSV 验证：单卡 tiny smoke 生成 `training_diagnostics.csv`，表头为 `global_step,loss,lr`，每个 logged optimizer step 都有记录。

## 已完成/可保留

- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:84) 的 cosine LR 实现正确，REPORT 中 LR 数值已更正。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:207) 新增 rank0 diagnostics CSV 初始化，默认路径为 `--output-dir/training_diagnostics.csv`。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:219) 新增 diagnostics 追加写入，列为 `global_step,loss,lr`。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:97) 的 `_load_config()` 已支持 preset name 和 YAML path 双模式。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:203) 已移除 meta-device 初始化，tiny FSDP2 单卡训练能实际进入 forward/backward。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:226) 当前使用顶层 `fully_shard(model)`，本轮 2-GPU/4-GPU tiny 均通过。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:232) 已添加 `--act-ckpt`，并导入 `Block`，64K preflight 报告显示开启 activation checkpointing 后 4-GPU 可跑。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:159) 的 RNG state 恢复已修复，`--resume auto` 在单卡和 2-GPU tiny 上通过。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:261) 已接入 map-style `ProLongPackedDataset`，batch 契约与 `labels=input_ids` 一致。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:269) 当前使用 `TensorDataset` + 手动 rank stride 分片，tiny 多卡不再卡住。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:246) 使用 bf16 autocast，符合 A100 训练方向。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:294) 已移除外层 `itertools.count()` 无限 epoch 循环，训练循环更简单。
- [train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:361) 已有 `finally: dist.destroy_process_group()`，失败/中断时清理路径比早期版本更好。
- [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:187) 的 `total_tokens` 和 [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:192) 的 `total_blocks` 已改为当前字段，不再抛 `AttributeError`。

## 用户决策

- 不保留 validation split。
- 正式训练只使用 `train_*.bin`。
- 当前 `/data3/junhaohu/data/prolong_64K_v2_subset` 已确认为 train-only：`948` 个 `train_*.bin`、`0` 个 `validation_*.bin`。

## 风险评估

1. 梯度裁剪被临时移除  
   REPORT 说明根因是 `clip_grad_norm_()` + FSDP2 多 rank 交互。这个风险**不阻塞正式训练启动**：当前 preflight 的 loss 正常下降，且 no-clip 通常可作为临时训练策略。建议正式训练时密切观察 `training_diagnostics.csv` 中 loss 是否出现 NaN/Inf 或异常尖峰；若发生不稳定，再优先实现 FSDP2 安全 grad norm/clip。

2. 当前 `preload=True` 方案不适合完整真实 ProLong  
   [prolong_dataset.py](/data3/junhaohu/comb/baselines/ArchScale/data/prolong_dataset.py:138) 会把 `.bin` payload concatenate 成 int64 buffer；[train.py](/data3/junhaohu/comb/baselines/ArchScale/training/train.py:269) 又把所有 block stack 成 tensor。这个风险**不阻塞当前正式训练启动**，前提是训练命令只指向 train-only 1/20 subset `/data3/junhaohu/data/prolong_64K_v2_subset`；不能指向完整 84GB 解压目录。

## 建议修复

- 正式训练使用 train-only subset 目录 `/data3/junhaohu/data/prolong_64K_v2_subset`，不要指向完整解压目录，避免整包 preload。
- 对比训练曲线时读取 `training_diagnostics.csv`；当前可用字段为 `global_step,loss,lr`。
- 保留本轮通过命令作为正式 smoke 标准：单卡、单卡 resume、2-GPU no-mid-ckpt、2-GPU ckpt、2-GPU resume、4-GPU ckpt。
- 梯度裁剪不阻塞本轮正式训练启动；训练期间需监控 loss，若出现数值异常再恢复 FSDP2 安全裁剪。

## 是否可以进入下一步

- **步骤 5：通过 synthetic smoke。**
- **步骤 6：通过真实数据校验与 4-GPU 64K preflight。**
- **可以开始正式训练。** 训练命令必须使用 train-only subset 目录，并带 `--act-ckpt`；训练 loss 曲线会写入 `training_diagnostics.csv`。
