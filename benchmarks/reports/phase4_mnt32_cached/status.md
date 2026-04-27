# Phase 4 Cached-Chunk Dev Sweep Status

## What Changed

The benchmark adapter now caches CombLlama chunk encoder outputs once per prompt:

```text
prompt history -> chunk_model -> cross_attention_states
```

During generation, the adapter reuses `cross_attention_states` and only re-runs the recent decoder window. This does not modify `models/CombLlama.py` and does not rely on the unverified self-attention KV cache path.

Config flag:

```json
"cache_chunk_states": true
```

CLI override:

```bash
--cache-chunk-states true|false
```

## Validation

Two equivalence checks were run with cache on/off:

```text
RULER 4K sample:
  output identical
  decode latency: 1.24s -> 0.75s

SCBench KV sample:
  output identical
  decode latency: 17.16s -> 1.64s
```

## Full Dev Sweep

The previously too-slow `max_new_tokens=32` full dev sweep now runs successfully.

```text
benchmarks: RULER, LongBench, SCBench, LongCodeBench, LoCoMo
compression_ratios: 1.0, 0.5, 0.25, 0.125
examples per ratio: 25
total prediction rows: 100
errors: 0
gpu: CUDA_VISIBLE_DEVICES=0
```

Outputs:

```text
benchmarks/results/phase4_combllama_dev_sweep_mnt32_cached/
benchmarks/reports/phase4_combllama_dev_sweep_mnt32_cached_summary.csv
benchmarks/reports/phase4_mnt32_cached/
```

## Aggregate Results

```text
ratio 1.0:   avg chunk tokens 23926.9, max memory 38.46 GB, avg prefill 0.568s, avg decode 3.345s, contains 0.20
ratio 0.5:   avg chunk tokens 11557.0, max memory 30.33 GB, avg prefill 0.320s, avg decode 3.269s, contains 0.08
ratio 0.25:  avg chunk tokens  5535.9, max memory 26.33 GB, avg prefill 0.205s, avg decode 3.082s, contains 0.08
ratio 0.125: avg chunk tokens  2709.6, max memory 24.33 GB, avg prefill 0.151s, avg decode 2.992s, contains 0.08
```

## Remaining Limitation

The adapter still does not use self-attention KV cache. It re-runs the recent decoder window for each generated token. This is now acceptable for dev sweeps, but official full-size evaluations should either:

1. implement verified self-attention KV cache reuse, or
2. keep generation length and sample count controlled.

## Recommended Next Step

Implement official benchmark-specific metrics and prompt formatting before expanding sample counts:

```text
RULER: exact passkey match by length and needle position
LongBench: official task metrics
SCBench: task-specific shared-context scoring
LongCodeBench: multiple-choice/code QA exact letter scoring
LoCoMo: MC10 exact-choice scoring
```

## Scoring Pass

A task-aware diagnostic scorer has been added:

```text
benchmarks/scripts/score_predictions.py
```

Scored outputs:

```text
benchmarks/reports/phase4_mnt32_cached/scored/scored_predictions.jsonl
benchmarks/reports/phase4_mnt32_cached/scored/task_summary.csv
benchmarks/reports/phase4_mnt32_cached/scored/benchmark_summary.csv
benchmarks/reports/phase4_mnt32_cached/scored/score_report.md
```

Current scoring rules:

```text
RULER: passkey containment
LongCodeBench: multiple-choice letter/text matching
LoCoMo: multiple-choice/text matching
SCBench: contains-match fallback
LongBench: contains-match fallback
```

Current dev scores:

```text
ratio 1.0: LoCoMo 1.000, LongBench 0.200, LongCodeBench 0.000, RULER 0.083, SCBench 0.000
ratio 0.5: all benchmark-level task-aware scores are 0.000 on this tiny dev subset
ratio 0.25: all benchmark-level task-aware scores are 0.000 on this tiny dev subset
ratio 0.125: all benchmark-level task-aware scores are 0.000 on this tiny dev subset
```

Interpretation:

These scores are useful for debugging the quality-efficiency trend, but not yet final benchmark scores. LongBench and SCBench still need their official evaluators, and the dev subset is intentionally tiny.
