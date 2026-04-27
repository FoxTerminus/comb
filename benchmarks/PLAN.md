# YOCO vs CombLlama Benchmark Plan

This directory stores the benchmark plan, adapters, configs, raw outputs, and reports for comparing YOCO and CombLlama as long-context KV-cache compression baselines.

## 1. Evaluation Goal

The comparison should answer four questions:

1. Does CombLlama preserve long-context information better than YOCO under the same training data and model scale?
2. Does CombLlama provide a better quality-efficiency trade-off when KV cache is compressed?
3. Which task types are most sensitive to cache compression: retrieval, reasoning, coding, or long dialogue memory?
4. At which context length and compression ratio does each model start to fail?

The evaluation must report both task quality and system efficiency. Accuracy without memory/latency numbers is insufficient for this project.

## 2. Directory Layout

```text
benchmarks/
  PLAN.md
  configs/          # model paths, decoding settings, context lengths, compression ratios
  scripts/          # shared runners, launch scripts, metric aggregation
  results/          # raw predictions, raw metrics, per-run logs
  reports/          # summarized tables, plots, final comparison notes
  RULER/            # effective-context-length benchmark
  LongBench/        # general long-context understanding benchmark
  SCBench/          # KV-cache-centric benchmark
  LongCodeBench/    # long-context coding benchmark
  LoCoMo/           # long-term multi-session dialogue benchmark
```

## 3. Model Matrix

Evaluate at least these systems:

1. `YOCO`: the current YOCO baseline initialized from Llama and trained with chunk text prepended as masked decoder context.
2. `CombLlama-full-cache`: CombLlama without KV compression, used as the upper bound for its architecture.
3. `CombLlama-compressed`: CombLlama with KV cache compression enabled.
4. Optional `Llama-full-context`: original Llama baseline if GPU budget allows.

For CombLlama compression, use a fixed compression sweep:

```text
compression_ratio = 1.0, 0.5, 0.25, 0.125
```

For YOCO, report the same context-length settings but no compression ratio unless a comparable cache budget is explicitly implemented.

## 4. Shared Evaluation Settings

Use identical decoding settings across models:

```text
temperature = 0.0
top_p = 1.0
max_new_tokens = task-specific
batch_size = task-specific, constrained by memory
dtype = bfloat16
gpus = 2,3,4,7 unless unavailable
```

Evaluate these context lengths when supported by the benchmark:

```text
4K, 8K, 16K, 32K, 64K, 128K
```

If a benchmark cannot naturally scale to all lengths, report its native length distribution and do not synthesize misleading padding unless the benchmark protocol requires it.

## 5. Metrics

Quality metrics:

1. Accuracy, exact match, F1, ROUGE, edit similarity, or pass rate according to each benchmark's official protocol.
2. Per-category scores, not only overall scores.
3. Length-bucketed scores when available.

Efficiency metrics:

1. Peak GPU memory.
2. Prefill latency.
3. Decode latency.
4. Tokens per second.
5. KV cache size.
6. Compression ratio.
7. Quality-efficiency Pareto curves.

Every result row should contain:

```text
model, checkpoint, benchmark, task, context_length, compression_ratio,
quality_metric, peak_memory_gb, prefill_latency_s, decode_latency_s,
tokens_per_second, run_time, git_commit, notes
```

## 6. Benchmark Roles

### 6.1 RULER

Role: controlled long-context stress test.

Tasks to include:

1. Retrieval tasks: test whether models can find target facts in long distractor contexts.
2. Multi-hop tracing tasks: test whether models can follow chains of references across long inputs.
3. Aggregation tasks: test whether models can combine multiple scattered facts.
4. QA tasks: test long-context question answering under controlled length.

What it measures:

1. Effective usable context length.
2. Lost-in-the-middle behavior.
3. Compression sensitivity under synthetic but controlled pressure.
4. Failure point as context length increases.

Primary outputs:

```text
score_by_context_length.csv
score_by_task_type.csv
compression_curve.csv
```

### 6.2 LongBench

Role: general real-world long-context understanding.

Tasks to include:

1. Single-document QA.
2. Multi-document QA.
3. Summarization.
4. Few-shot learning.
5. Synthetic retrieval tasks.
6. Code completion tasks included in LongBench.

What it measures:

1. General long-context comprehension.
2. Robustness across natural-language task families.
3. Whether improvements on RULER transfer to realistic tasks.
4. Basic long-code capability through the code-related LongBench subset.

Primary outputs:

```text
longbench_overall.csv
longbench_by_task.csv
longbench_by_length.csv
```

### 6.3 SCBench

Role: main KV-cache compression benchmark.

Tasks to include:

1. String retrieval.
2. Semantic retrieval.
3. Global information understanding.
4. Multi-task shared-context evaluation.

What it measures:

1. KV cache generation cost.
2. KV cache compression quality.
3. Retrieval from compressed cache.
4. Shared-context reuse efficiency.
5. Quality-efficiency trade-off under repeated queries over the same long context.

Primary outputs:

```text
scbench_quality.csv
scbench_cache_memory.csv
scbench_latency.csv
scbench_pareto.csv
```

### 6.4 LongCodeBench

Role: dedicated long-context coding benchmark.

Tasks to include:

1. LongCodeQA: repository-level code understanding and code QA.
2. LongSWE-Bench: long-context bug fixing and code repair.

What it measures:

1. Cross-file code understanding.
2. Repository-level dependency tracking.
3. Long-range API and symbol resolution.
4. Bug localization and patch generation.
5. Whether KV cache compression preserves important code definitions and call chains.

Primary outputs:

```text
longcodebench_qa.csv
longcodebench_repair.csv
longcodebench_by_repo_size.csv
```

Fallback if LongCodeBench integration is blocked:

1. Use RepoBench for repository-level retrieval/completion/pipeline evaluation.
2. Use CrossCodeEval for multilingual cross-file code completion.

### 6.5 LoCoMo

Role: dedicated long-dialogue and long-term memory benchmark.

Tasks to include:

1. Text-only QA.
2. Single-hop conversation memory.
3. Multi-hop cross-session reasoning.
4. Temporal reasoning.
5. Adversarial or unanswerable questions.
6. Optional event summarization after QA is stable.

What it measures:

1. Long-term conversational memory.
2. Multi-session fact recall.
3. Temporal ordering and event consistency.
4. Robustness to irrelevant dialogue history.
5. Whether cache compression damages early-session facts.

Preferred protocol:

1. Start with LoCoMo QA text-only.
2. Add LoCoMo-MC10 if deterministic multiple-choice accuracy is needed.
3. Do not use multimodal dialogue generation unless both YOCO and CombLlama support the same image or caption input format.

Primary outputs:

```text
locomo_qa.csv
locomo_by_question_type.csv
locomo_temporal.csv
locomo_mc10.csv
```

## 7. Execution Phases

### Phase 0: Local Infrastructure

Tasks:

1. Create shared model loading adapters for YOCO and CombLlama.
2. Define a common generation API: `generate(prompt, max_new_tokens, compression_ratio)`.
3. Add timing and memory instrumentation around prefill and decode.
4. Add config files for model paths, GPU mapping, dtype, and benchmark-specific settings.
5. Add result schema validation.

Exit criteria:

1. One synthetic prompt can run on YOCO and CombLlama.
2. Results include quality placeholder, memory, latency, and generation text.
3. Runs never use GPU 0 or GPU 1 unless explicitly overridden.

### Phase 1: Smoke Evaluation

Tasks:

1. Run 5 to 20 examples per benchmark.
2. Verify input formatting, max length handling, and decoding termination.
3. Verify each metric script can parse predictions.
4. Verify result files are written under `benchmarks/results/`.

Exit criteria:

1. All five benchmarks produce at least one valid result file.
2. No benchmark crashes on either YOCO or CombLlama.
3. Peak memory and latency are recorded for every example.

### Phase 2: Development Subset

Tasks:

1. Run RULER at 4K, 8K, 16K, and 32K.
2. Run LongBench dev subset.
3. Run SCBench on a small shared-context subset.
4. Run LongCodeBench QA subset.
5. Run LoCoMo QA or LoCoMo-MC10 subset.

Exit criteria:

1. Results are stable across two repeated runs within expected deterministic tolerance.
2. CombLlama full-cache and compressed modes both run.
3. YOCO baseline runs with the same prompts and decoding settings.
4. Initial summary tables are generated under `benchmarks/reports/`.

### Phase 3: Full Evaluation

Tasks:

1. Run the full benchmark matrix.
2. Sweep context lengths where supported.
3. Sweep CombLlama compression ratios.
4. Save raw predictions and raw metrics.
5. Generate aggregate tables and plots.

Exit criteria:

1. Main report includes per-benchmark and per-task results.
2. Efficiency metrics are available for every model and compression setting.
3. Results are reproducible from config files and scripts.

### Phase 4: Analysis

Tasks:

1. Compare YOCO vs CombLlama at equal context length.
2. Compare CombLlama full-cache vs compressed cache.
3. Identify the compression ratio where quality drops sharply.
4. Analyze task categories where compression helps, hurts, or has no effect.
5. Produce final Pareto plots: score vs memory and score vs latency.

Exit criteria:

1. Final report explains not only which model wins, but why.
2. Failure cases are grouped by benchmark and task type.
3. Claims are backed by both quality metrics and efficiency metrics.

## 8. Recommended First Runs

Start with small deterministic runs:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,7 PYTHONPATH=/data3/junhaohu/comb \
python benchmarks/scripts/run_smoke.py \
  --models yoco,combllama \
  --benchmarks ruler,longbench,scbench,longcodebench,locomo \
  --num-examples 10 \
  --output-dir benchmarks/results/smoke
```

Then run the development subset:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,7 PYTHONPATH=/data3/junhaohu/comb \
python benchmarks/scripts/run_dev_subset.py \
  --config benchmarks/configs/dev.yaml \
  --output-dir benchmarks/results/dev
```

The scripts above are planned interfaces. They should be implemented in Phase 0.

## 9. Risks and Controls

Risk: benchmarks use different prompt formats.
Control: store rendered prompts and raw generations for every example.

Risk: compression improves memory but damages early-context recall.
Control: report length buckets and question types, especially RULER retrieval and LoCoMo temporal/multi-hop.

Risk: coding benchmarks require repository setup and may be slow.
Control: start with LongCodeQA before LongSWE-Bench repair tasks.

Risk: generation metrics are noisy.
Control: use deterministic decoding and official metrics whenever possible.

Risk: GPU memory measurements are inconsistent.
Control: reset CUDA memory stats per example and record peak allocated/reserved memory.

Risk: benchmark contamination.
Control: avoid using training data that overlaps with benchmark test sets where possible and report benchmark versions.

## 10. Final Deliverables

1. `benchmarks/configs/*.json`: reproducible run configs.
2. `benchmarks/scripts/*.py`: benchmark runners and aggregators.
3. `benchmarks/results/current/`: current raw predictions and manifest.
4. `benchmarks/reports/current_benchmark_summary.csv`: benchmark-level aggregate table.
5. `benchmarks/reports/current_task_summary.csv`: task-level aggregate table.
6. `benchmarks/reports/current_failure_summary.csv`: collapse and packing diagnostic summary.
7. `benchmarks/reports/final_report.md`: consolidated current report.
8. Older phase outputs are removed after consolidation so the tree keeps only the latest result set.

## 11. Current Implementation Progress

The benchmark harness is currently focused on CombLlama because YOCO is still training.

Completed:

1. Local schemas, prompt rendering, runners, scoring, sweep, diagnostics, and summary scripts are implemented under `benchmarks/scripts/`.
2. Dev subsets exist for RULER, LongBench, SCBench, LongCodeBench, and LoCoMo.
3. CombLlama dev sweeps have been validated with chunk-state caching and deterministic generation.
4. Chunk-dropping retention policies were removed from the runnable benchmark path.
5. Phase 13 redefined the main CombLlama run so all chunks are sent through the encoder and no evidence is dropped by the benchmark harness.
6. The main CombLlama-vs-YOCO comparison should use `all_encoder_chunks`.

Current default:

```json
"retention_policy": "all_encoder_chunks"
```

Latest status:

1. The consolidated current report is `benchmarks/reports/final_report.md`.
2. The current raw predictions and manifest are under `benchmarks/results/current/`.
3. Phase 15 ran 50 examples per benchmark under `all_encoder_chunks`: 250 successful generations, 0 errors, 0 dropped chunks, and 0 detected collapse.
4. Historical phase outputs were removed after consolidation to reduce redundancy.
5. The current recommendation is `all_encoder_chunks` for fair CombLlama-vs-YOCO comparison.
6. The next implementation step should integrate official evaluators or rebalance each benchmark's 50-example subset across task families before final comparison.
