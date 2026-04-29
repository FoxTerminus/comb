# 从零重建 CombLlama Benchmark 体系计划

## 1. 总览

目标目录：`/data3/junhaohu/comb/benchmarks`。

本目录已经按“统一框架优先”的方案从零重建。评测对象为 `CombLlama`、原始 `Llama`、`YOCO`、`SambaY`；主要 benchmarks 为 `RULER`、`SCBench`、`LongBench`，次要 benchmarks 为 `LoCoMo`、`LongCodeBench`。默认运行环境为单机多卡，模型串行、任务可分片。

## 2. 固定目录与接口

```text
benchmarks/
  PLAN.md
  README.md
  configs/
    models/
    runs/
    benchmarks/
  data/
    raw/
    processed/
    smoke/
    dev/
  scripts/
    adapters/
    runners/
    scoring/
    reporting/
    utils/
    converters/
  results/
    smoke/
    dev/
    full/
  reports/
  logs/
  tests/
```

统一样本格式为 `BenchmarkExample`，统一预测格式为 `GenerationRecord`。所有 adapter 暴露：

```python
generate(example, generation_config) -> GenerationRecord
```

每条预测必须包含 `error` 字段；单样本失败不得中断整次 run。

## 3. 阶段计划

### Phase 0：清空与骨架

- 删除旧 `benchmarks` 内容。
- 新建目录骨架、`PLAN.md`、`README.md`。
- 建立 schema、配置读取、日志、结果写入、随机种子、run id、环境记录。
- 添加 mock adapter，保证不加载模型也能跑通全链路。

验收命令：

```bash
python -m benchmarks.scripts.runners.run_smoke --model mock
```

### Phase 1：统一数据层

- 每个 benchmark 拥有独立 converter。
- `smoke` 默认使用可离线运行的小样本。
- `dev` 默认由 smoke 扩展生成，后续可替换为官方 dev 子集。
- `full` 只从 `data/raw` 转换，不改写官方原始文件。
- 所有样本必须通过 schema 校验，并记录 `context_length`、`source`、`metric`。

### Phase 2：模型适配层

- `LlamaAdapter` 使用 `transformers.AutoModelForCausalLM`。
- `YOCOAdapter` 使用 `baselines.YOCO.models.YOCO.YOCOForCausalLM`。
- `SambaYAdapter` 使用 `baselines.SambaY.models.SambaY.SambaYForCausalLM`。
- `CombLlamaAdapter` 使用 `models.CombLlama.CombLlamaForConditionalGeneration`。
- checkpoint 不完整时必须记录清晰错误。

### Phase 3：运行器与报告

统一运行命令：

```bash
python -m benchmarks.scripts.runners.run_eval \
  --run-config benchmarks/configs/runs/dev_combllama.json
```

每次运行输出：

```text
results/{smoke|dev|full}/{run_id}/
  predictions.jsonl
  metrics.json
  failures.jsonl
  run_config.resolved.json
  environment.json
```

报告输出到 `reports/`，包含 benchmark 汇总、模型对比、失败汇总和 CombLlama Pareto 数据。

## 4. 指标

- `RULER`：exact match、contains match、按 context length / task type 汇总。
- `SCBench`：exact/contains、cache reuse metadata、shared context 多 query 汇总。
- `LongBench`：F1、Rouge-L、classification accuracy。
- `LoCoMo`：QA F1、exact match、question type、temporal 分组。
- `LongCodeBench`：edit similarity、exact match、代码任务官方指标预留。
- 效率指标：peak memory、prefill latency、decode latency、tokens/s、prompt/context/generated tokens、架构固定的 `kv_cache_policy`、CombLlama 的 `chunk_size` / `recent_window_tokens`、错误数。

## 5. 最终验收

- `mock` 模式完整通过 smoke/dev。
- `CombLlama` 能通过 smoke，若 checkpoint 或 CUDA 不可用则生成明确失败记录。
- 至少一个 baseline 能通过 smoke，若环境未就绪则 failure summary 明确原因。
- dev 运行能生成 per-model、per-benchmark、per-task 汇总。
- 所有结果可从 `run_config.resolved.json` 复现。
- `RULER`、`SCBench`、`LongBench` 标记为 primary 并在报告中优先展示。
- `LoCoMo`、`LongCodeBench` 标记为 secondary，不阻塞 primary 报告生成。

## 6. KV Cache Policy 约束

不要把“压缩率”当作可调模型参数写进配置。当前模型的 KV cache 压缩方式来自架构本身：

- `Llama`：完整 decoder-only KV cache，作为无压缩参考线。
- `CombLlama`：用层数更少的 chunk encoder 处理旧上下文，生成供 decoder cross-attention 使用的 K/V；近期 token 仍走 decoder self-attention。
- `YOCO`：self-decoder 生成可复用 memory/KV，cross-decoder 复用一份 cross-decoder-level K/V，而不是每个 cross-decoder layer 独立生成完整历史 KV。
- `SambaY`：使用自己的 hybrid decoder state/cache 路径，不复用 CombLlama 的 `chunk_ids -> chunk_model -> cross_attention_states`。

报告中只能记录这些固定 policy 以及真实运行参数，不做虚假的压缩率扫描。如果将来需要量化压缩收益，应从实际 cache/state 张量规模统计得到，而不是预设一个配置字段。
