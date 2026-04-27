# Phase 10: keep_even Fair Sweep on Core Long-Context Benchmarks

## Scope

This phase validates the fair, non-oracle `keep_even` retention policy on the currently prepared CombLlama dev subsets:

- RULER: 12 passkey retrieval examples across 4K/8K/16K/32K and early/middle/late answer positions.
- SCBench: 4 examples covering KV retrieval, repo/code QA, English QA, and summarization.
- LongBench: 5 examples covering Qasper, HotpotQA, GovReport, LCC, and RepoBench-P.

LongCodeBench and LoCoMo are intentionally excluded from this phase because the previous stages showed that the core failure mode should first be stabilized on retrieval, QA, summary, and code-context tasks.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,SCBench,LongBench \
  --compression-ratios 1.0,0.5,0.25 \
  --retention-policies keep_even \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase10_keep_even_ruler_scbench_longbench
```

Scoring and diagnostics:

```bash
PYTHONPATH=/data3/junhaohu/comb /data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/score_predictions.py \
  --manifest benchmarks/results/phase10_keep_even_ruler_scbench_longbench/dev_sweep_manifest.json \
  --split dev \
  --output-dir benchmarks/reports/phase10_keep_even/scored

PYTHONPATH=/data3/junhaohu/comb /data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/diagnose_failures.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,SCBench,LongBench \
  --compression-ratios 1.0,0.5,0.25 \
  --retention-policies keep_even \
  --predictions-manifest benchmarks/results/phase10_keep_even_ruler_scbench_longbench/dev_sweep_manifest.json \
  --output-dir benchmarks/reports/phase10_keep_even/failure_analysis
```

## Outputs

- Predictions: `benchmarks/results/phase10_keep_even_ruler_scbench_longbench/`
- Scored predictions: `benchmarks/reports/phase10_keep_even/scored/scored_predictions.jsonl`
- Benchmark summary: `benchmarks/reports/phase10_keep_even/scored/benchmark_summary.csv`
- Task summary: `benchmarks/reports/phase10_keep_even/scored/task_summary.csv`
- Failure diagnostics: `benchmarks/reports/phase10_keep_even/failure_analysis/`

## Results

| Compression | Benchmark | Score | Max Memory GB | Errors | Collapse Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 | RULER | 0.167 | 27.01 | 0 | 0.000 |
| 1.0 | SCBench | 0.119 | 38.46 | 0 | 0.000 |
| 1.0 | LongBench | 0.140 | 23.11 | 0 | 0.000 |
| 0.5 | RULER | 0.083 | 24.62 | 0 | 0.000 |
| 0.5 | SCBench | 0.136 | 30.33 | 0 | 0.000 |
| 0.5 | LongBench | 0.066 | 22.71 | 0 | 0.000 |
| 0.25 | RULER | 0.167 | 23.41 | 0 | 0.000 |
| 0.25 | SCBench | 0.074 | 26.33 | 0 | 0.000 |
| 0.25 | LongBench | 0.066 | 22.62 | 0 | 0.000 |

## Interpretation

`keep_even` fixes the generation-collapse problem observed with `keep_recent`: all tested benchmark/ratio combinations have `collapse_rate=0.0` and `num_errors=0`.

The remaining quality bottleneck is evidence retention rather than decoding stability. RULER diagnostics show answer-dropping rates of `0.583` at ratio `0.5` and `0.667` at ratio `0.25`; LongBench also loses evidence on part of the compressed samples. This explains why compressed scores remain low even though the model no longer degenerates.

Memory decreases with compression, especially on SCBench: max memory drops from `38.46 GB` at ratio `1.0` to `30.33 GB` at ratio `0.5` and `26.33 GB` at ratio `0.25`. This confirms the benchmark harness can measure the expected memory-quality tradeoff.

## Next Step

Phase 11 should add a fair answer-agnostic retention policy that is stronger than uniform even sampling. The most useful candidate is a mixed policy such as `keep_early_recent_even`, which reserves some chunks from the beginning, some from the end, and distributes the rest evenly through the middle. That should preserve instruction/task setup, recent query context, and long-range evidence better than pure `keep_even`, while remaining fair for comparison with YOCO.
