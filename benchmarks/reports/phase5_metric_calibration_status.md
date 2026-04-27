# Phase 5 Metric Calibration Status

## Scope

This phase recalibrates scoring for the existing CombLlama dev sweep without
running model inference again. It uses the cached predictions from:

```text
benchmarks/results/phase4_combllama_dev_sweep_mnt32_cached/dev_sweep_manifest.json
```

## What Changed

The lightweight metric library now includes:

```text
QA/token F1
Rouge-L F1
edit similarity
normalized exact match
normalized contains match
```

The task-aware scorer now maps benchmark task families to more appropriate
metrics:

```text
RULER: passkey containment
LongBench QA: QA F1
LongBench summarization: Rouge-L F1
LongBench code tasks: edit similarity
LongBench classification/retrieval: exact match
SCBench summary: Rouge-L F1
SCBench repo/code QA: edit similarity
SCBench KV retrieval: exact-or-contains
SCBench open QA: QA F1
LongCodeBench: multiple-choice matching
LoCoMo MC10: multiple-choice matching
```

These are local near-official metric families for phase gating. Final paper or
report numbers should still be regenerated with the upstream official
evaluators once the full benchmark matrix is ready.

## Validation

Static checks passed:

```bash
/data3/junhaohu/anaconda3/envs/comb/bin/python -m py_compile \
  benchmarks/scripts/metrics.py \
  benchmarks/scripts/score_predictions.py
```

Scoring command:

```bash
PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/score_predictions.py \
  --manifest benchmarks/results/phase4_combllama_dev_sweep_mnt32_cached/dev_sweep_manifest.json \
  --split dev \
  --output-dir benchmarks/reports/phase5_metric_calibration
```

## Outputs

```text
benchmarks/reports/phase5_metric_calibration/scored_predictions.jsonl
benchmarks/reports/phase5_metric_calibration/task_summary.csv
benchmarks/reports/phase5_metric_calibration/benchmark_summary.csv
benchmarks/reports/phase5_metric_calibration/score_report.md
```

## Current Dev Scores

```text
ratio 1.0:
  LoCoMo 1.000
  LongBench 0.154
  LongCodeBench 0.000
  RULER 0.083
  SCBench 0.104

ratio 0.5:
  LongBench 0.042
  SCBench 0.006
  others 0.000

ratio 0.25:
  LongBench 0.040
  SCBench 0.004
  others 0.000

ratio 0.125:
  LongBench 0.040
  SCBench 0.007
  others 0.000
```

## Interpretation

The recalibrated scores preserve the same high-level trend as Phase 4: the
current tiny dev subset shows a clear quality drop when moving away from
`compression_ratio=1.0`, while memory improves substantially. The new metrics
are more informative for partial answers, especially LongBench QA, LongBench
code completion, and SCBench summaries.

## Recommended Next Step

Before scaling sample counts, the next phase should improve prompt templates
per benchmark family:

```text
RULER: require answer-only passkey output
LongBench QA: answer-only short form
LongBench summarization: summary-specific max_new_tokens
LongBench/SCBench code: preserve code-only completion format
LongCodeBench/LoCoMo: require one option letter only
```

After prompt calibration, rerun the same tiny dev sweep so quality changes can
be attributed to prompt formatting rather than sample-size changes.
