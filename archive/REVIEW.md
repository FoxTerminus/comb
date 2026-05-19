# REVIEW: Comb 64K-Prefix / 32K-Target Preflight Gate

日期: 2026-05-01

审核对象: `PLAN.md`、`REPORT.md` 最新版、`data/prolong_qwen_dataset.py`、`training/train.py`、`scripts/run_comb_train.sh`、`models/comb_qwen.py`、旧版 `archive/.../data/base.py`

## 结论

**可以开始正式训练。**

这轮从头到尾复核后，当前实现已经对齐用户决策的 **块级前缀记忆方案**:

- 一个训练样本是 `prefix_len + target_len = 65536 + 32768 = 98304` tokens。
- `prefix` 只进入 chunk encoder，构造 cross-attention memory。
- `target` 只进入 decoder，并只在 target 内做 next-token loss。
- Dataset 使用 `stride = target_len = 32768` 滑窗，能够凑够 5913 steps。
- Qwen3 decoder self-attention 已统一 patch 到 SDPA causal 路径，`from_scratch=True/False` 都生效。

此前 `loss.backward()` 阶段的 full logits OOM 已修复。最新代码使用 token-chunked loss，不再 materialize 完整 `[1, 32768, 151936]` fp32 logits。真实 4 卡 64K+32K preflight 已通过 2 step，并额外验证了 checkpoint save 和 resume。

正式训练可使用 `scripts/run_comb_train.sh` 启动。建议启动后重点观察前 10 step 的 loss、吞吐和显存，再让它长期跑。

## 关键通过项

### 数据构造

位置: `training/train.py:184-190`, `training/train.py:231-264`

当前训练脚本:

```python
window_len = args.ctx_len + args.target_len
stride = args.target_len
```

每个 window 内:

```python
prefix = full_block[:, :prefix_len]
target = full_block[:, prefix_len:]
chunk_ids = prefix.view(1, n_chunks, chunk_size).reshape(1, -1)
input_ids = target
```

这避免了之前 `chunk_ids=input_ids` 的未来泄漏问题。现在 cross-attention memory 只来自 target 之前的 prefix，decoder loss 只监督 target 段。

### Sliding window step 数

位置: `data/prolong_qwen_dataset.py:95-99`

Dataset 按 `start += stride` 生成窗口。实测当前数据:

```text
total tokens: 1,705,612,502
window size: 98,304
stride: 32,768
windows: 49,726
4-GPU per-rank windows: about 12,431
target-token budget: about 1.63B
sample shape: torch.Size([98304])
```

因此 4 卡训练 5913 optimizer steps 有足够样本。

### 旧 CombLlama 对齐

旧 collator 中 decoder position 和 chunk position 都是局部从 0 开始:

- `archive/.../data/base.py:54`: `position_ids = arange(len(q_ids))`
- `archive/.../data/base.py:65`: `position_ids_k = arange(len(c_ids))`

当前实现:

- target decoder position: `0..target_len-1`
- 每个 prefix chunk position: `0..chunk_size-1`
- `cu_seqlens_chunk` 把 64 个 prefix chunks 隔开做块内 self-attention

这与旧 CombLlama 的块级记忆范式一致。

### 因果性修复

位置: `models/comb_qwen.py:418-437`

`CombForConditionalGeneration.__init__()` 现在无论是否 `from_scratch` 都调用:

```python
self._fix_causal_mask()
```

并把所有 Qwen3 decoder self-attention 设置为:

```python
layer.self_attn.config._attn_implementation = "sdpa"
```

实测 `configs/comb_tiny.yaml + --no-from-scratch`:

```text
attn_impl ['sdpa', 'sdpa', 'sdpa', 'sdpa']
prefix_logits_max_diff 0.0
future_logits_max_diff 2.55078125
```

测试含义: 固定 prefix，扰动 target 位置 20 之后的 token，位置 0..19 的 logits 完全不变，说明 decoder 不看未来 token；未来位置 logits 改变是预期现象。

### Smoke test

GPU 6 上完成 1-step tiny smoke:

```text
CUDA_VISIBLE_DEVICES=6
config=configs/comb_tiny.yaml
ctx_len=64
target_len=32
total_steps=1
--no-from-scratch

Model: 592.6M total, 592.6M trainable
Sliding window: 96 tokens (prefix=64, target=32, stride=32)
step 1/1 | loss: 12.2616 | lr: 1.00e-05 | mem: 9.5GB
```

本轮 smoke 没有再出现 `_attn_implementation=None` warning。

`py_compile` 也已通过:

```text
models/comb_qwen.py
training/train.py
data/prolong_qwen_dataset.py
```

## 已关闭阻塞问题

### P0: full target logits 在 backward 阶段 OOM（已修复）

位置: `models/comb_qwen.py:40-47`, `models/comb_qwen.py:405-409`, `training/train.py:259-273`

旧 loss 路径:

```python
logits = _slice_logits(self.lm_head, outputs.last_hidden_state, 0)
return logits.float()
loss = CrossEntropyLoss(logits.view(-1, vocab_size), shift_labels.view(-1))
```

正式配置下:

```text
target_len = 32768
vocab_size = 151936
logits shape = [1, 32768, 151936]
```

仅 logits 张量大小:

```text
bf16: 32768 * 151936 * 2 bytes ≈ 9.28 GiB
fp32: 32768 * 151936 * 4 bytes ≈ 18.55 GiB
```

旧实现由于 `_slice_logits()` 强制 `logits.float()`，实际 loss/backward 会走 fp32 full logits 路径。

当前修复:

```python
def _chunked_loss(lm_head, hidden_states, shift_labels, chunk_size=512):
    ...
    logits_chunk = lm_head(hs_chunk).float()
    ...
    return total_loss / total_tokens.clamp(min=1)
```

`CombForCausalLM.forward()` 在有 `shift_labels` 时只返回 scalar loss，`logits=None`，避免持有完整 vocab logits。

### 历史 OOM 记录

测试命令核心配置:

```text
CUDA_VISIBLE_DEVICES=0,1,5,6
world_size=4
ctx_len=65536
target_len=32768
total_steps=1
act_ckpt=True
```

已通过阶段:

```text
World: 4, GPUs: 0,1,5,6, seed: 42
Model: 1127.4M total, 220.2M trainable
ProLongQwenDataset: 948 files, 1.71B tokens, 49726 blocks of 98304
Sliding window: 98304 tokens (prefix=65536, target=32768, stride=32768)
Total windows: 49726, per rank: 12431, target budget: 1.63B tokens
Required steps: 1, available per rank: 12431
```

失败点:

```text
training/train.py:273
loss.backward()

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.55 GiB.
process has about 64.6 GiB memory in use
about 14.5 GiB free
```

这个 `18.55 GiB` 与完整 fp32 logits 的理论大小完全对齐。因此 OOM 不是数据构造、权重加载、FSDP 初始化或 FlashAttention forward 首先导致，而是 full vocab logits/loss backward 路径导致。

修复后预期峰值:

```text
chunk=1024 时 fp32 logits 峰值约 1024 * 151936 * 4 bytes ≈ 0.58 GiB
chunk=512  时 fp32 logits 峰值约 0.29 GiB
```

修复后已重跑真实 4 卡 preflight，见下一节。

## 最新验证

### 64K+32K 2-step preflight

输出目录: `/data3/junhaohu/checkpoints/Comb-Qwen3-1B-Prolong-preflight`

CSV:

```text
global_step,loss,lr
1,9.953794479370117,0.0002999999999999999
2,10.574728012084961,2.9999999999999997e-05
```

REPORT 记录的日志:

```text
step 1/2 | loss: 9.9538 | lr: 3.00e-04 | tok/s: 2974 | mem: 30.3GB
step 2/2 | loss: 10.5747 | lr: 3.00e-05 | tok/s: 4194 | mem: 30.3GB
```

通过项:

- 真实 `ctx_len=65536,target_len=32768`。
- 4 卡 world size。
- loss 有限，无 NaN/Inf。
- 峰值显存约 30.3GB，80GB 卡余量充足。
- CSV 正常写入。
- windows 数为 49,726，4 卡 per-rank 约 12,431，足够 5,913 step。

### 64K+32K 10-step preflight

输出目录: `/tmp/comb_preflight_10step`

运行配置:

```text
CUDA_VISIBLE_DEVICES=0,1,5,6
ctx_len=65536
target_len=32768
total_steps=10
act_ckpt=True
chunked loss chunk_size=512
```

完整 CSV:

```text
global_step,loss,lr
1,9.953794479370117,0.0002999999999999999
2,10.574728012084961,0.0002918585038060976
3,9.770132064819336,0.00026841599982106197
4,9.306068420410156,0.00023249999999999996
5,9.504310607910156,0.0001884425039850356
6,9.210917472839355,0.0001415574960149644
7,8.962654113769531,9.750000000000001e-05
8,6.430564880371094,6.158400017893797e-05
9,8.202664375305176,3.814149619390237e-05
10,8.10108757019043,2.9999999999999997e-05
```

日志摘要:

```text
step 1/10  | loss: 9.9538  | mem: 30.3GB
step 2/10  | loss: 10.5747 | mem: 30.3GB
step 3/10  | loss: 9.7701  | mem: 30.3GB
step 4/10  | loss: 9.3061  | mem: 30.3GB
step 5/10  | loss: 9.5043  | mem: 30.3GB
step 6/10  | loss: 9.2109  | mem: 30.3GB
step 7/10  | loss: 8.9627  | mem: 30.3GB
step 8/10  | loss: 6.4306  | mem: 30.3GB
step 9/10  | loss: 8.2027  | mem: 30.3GB
step 10/10 | loss: 8.1011  | mem: 30.3GB
```

结论:

- 10 step 全部完成，exit code 0。
- 峰值显存始终稳定在 30.3GB，没有逐步爬升。
- loss 全部有限，无 NaN/Inf。
- step 2 之后吞吐稳定约 4.1K-4.2K target tokens/s。
- 预检结束后 GPU `0,1,5,6` 释放正常。

这比 2-step preflight 更充分地确认了 chunked loss、activation checkpoint、FSDP、数据滑窗和 CSV 写入在短程训练内稳定。

### Checkpoint save 探针

测试命令使用 `--save-interval 1`，输出目录 `/tmp/comb_save_probe`。

结果:

```text
step 1/1 | loss: 9.9538 | lr: 3.00e-05 | tok/s: 3146 | mem: 30.3GB
Checkpoint: /tmp/comb_save_probe/step_000001.pt
```

文件:

```text
step_000001.pt 1568358158 bytes
training_diagnostics.csv 65 bytes
```

这说明当前 FSDP/DTensor 状态下 rank0 checkpoint 保存可以完成。

### Final checkpoint 探针

DeepSeek 已更新训练结束保存逻辑:

```python
if global_step % args.save_interval != 0:
    ckpt_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
    save_checkpoint(model, optimizer, global_step, config, args, rank, ckpt_path)
```

我用真实 4 卡 64K+32K 配置验证了 `total_steps=2, save_interval=999`，即最后一步不是保存间隔倍数的情况。

结果:

```text
step 1/2 | loss: 9.9538  | mem: 30.3GB
step 2/2 | loss: 10.5747 | mem: 30.3GB
Final checkpoint: /tmp/comb_final_save_probe/step_000002.pt
```

文件:

```text
step_000002.pt 1568358158 bytes
training_diagnostics.csv 109 bytes
```

结论: 正式训练 `total_steps=5913, save_interval=500` 时，除 `500,1000,...,5500` 外，结束时还会保存 `step_005913.pt`。此前“不会保存最终 step”的风险已关闭。

### Resume 探针

使用 `/tmp/comb_save_probe/step_000001.pt` 做 `--resume auto --total-steps 2`。

结果:

```text
Resumed from step 1
step 2/2 | loss: 18.9861 | lr: 3.00e-05 | tok/s: 3560 | mem: 30.3GB
```

结论: checkpoint 可以被当前 4 卡同 world-size 训练脚本加载并继续训练。

注意: 当前 checkpoint 不保存 DataLoader/iterator 进度，resume 后会从 shuffled rank subset 开头重新取样，因此中断恢复会重复已见过的一小段 window。对 5,913 step 长训不是硬阻塞，但如果非常在意精确续训，应后续补 sampler/step offset。

## 非阻塞问题

### P2: `REPORT.md` 仍有历史残留

`REPORT.md` 前半部分仍保留早期状态，例如 `chunk_ids=input_ids`、旧 64K+64K 方案、历史修复记录等。末尾最新结论是对的，但文档整体容易误读。

建议 DeepSeek 后续把 `REPORT.md` 顶部加一个“当前最终状态”摘要，历史过程放到附录。

### P2: `--target-len` 默认值仍是 65536

位置: `training/train.py:307-309`

正式脚本显式传入 `--target-len 32768`，所以不阻塞 preflight。但既然主实验已定为 64K+32K，建议把默认值同步成 32768，减少手工命令误跑。

### P2: `configs/comb_qwen_1b.yaml` 的 `max_position_embeddings=65536`

当前 decoder target position 是 `0..32767`，chunk 内 position 是 `0..1023`，所以 65536 不会限制当前 64K-prefix/32K-target 设计。若未来要改成 decoder 直接处理 64K 以上绝对位置，再重新评估 RoPE/position 上限。

## 正式训练建议

可直接启动:

```bash
bash /data3/junhaohu/comb/scripts/run_comb_train.sh
```

启动后前 10 step 观察:

- 峰值显存应接近 30GB，不应突然回到 60GB+。
- loss 有限，不是 `nan/inf`。
- 日志打印 `window=98304, prefix=65536, target=32768, stride=32768`。
- `training_diagnostics.csv` 持续写入。
- 没有 `_attn_implementation=None` warning。

正式训练仍建议保留 `save_interval=500`。已验证 `save_interval=1` 可保存，checkpoint 约 1.57GB。
