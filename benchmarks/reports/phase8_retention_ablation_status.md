# Phase 8 Retention Policy Ablation Status

## Scope

This phase tests whether compressed-context failure is caused mainly by the
current chunk retention policy. It uses a small RULER + SCBench subset only.

Run scope:

```text
benchmarks: RULER, SCBench
examples: 3 per benchmark, 6 total
compression_ratio: 0.5
retention_policies: keep_recent, keep_early, keep_even, oracle_keep_answer
max_new_tokens: 16
gpu: CUDA_VISIBLE_DEVICES=0
```

`oracle_keep_answer` is a diagnostic-only policy. It is not a fair benchmark
setting because it uses the gold answer location to choose retained chunks.

## What Changed

`benchmarks/scripts/prompting.py` now supports these retention policies:

```text
keep_recent
keep_early
keep_even
oracle_keep_answer
```

`benchmarks/scripts/run_sweep.py` now accepts:

```bash
--retention-policies keep_recent,keep_early,keep_even,oracle_keep_answer
```

`benchmarks/scripts/score_predictions.py` now groups summaries by
`retention_policy`, so multiple policies are not mixed in one score row.

`benchmarks/scripts/diagnose_failures.py` now records actual retained chunk
indices and uses those indices to classify whether the gold answer is in
`kept_chunk`, `decoder`, or `dropped_chunk`.

## Commands

Mock plumbing check:

```bash
PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER \
  --limit-per-benchmark 1 \
  --compression-ratios 0.5 \
  --retention-policies keep_recent,keep_early,keep_even,oracle_keep_answer \
  --max-new-tokens 4 \
  --mock \
  --output-dir benchmarks/results/phase8_mock_retention_check
```

Real ablation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,SCBench \
  --limit-per-benchmark 3 \
  --compression-ratios 0.5 \
  --retention-policies keep_recent,keep_early,keep_even,oracle_keep_answer \
  --max-new-tokens 16 \
  --output-dir benchmarks/results/phase8_retention_ablation_ruler_scbench
```

## Outputs

```text
benchmarks/results/phase8_mock_retention_check/
benchmarks/results/phase8_retention_ablation_ruler_scbench/
benchmarks/reports/phase8_retention_ablation/scored/
benchmarks/reports/phase8_retention_ablation/failure_analysis/
```

## Scores

Benchmark summary:

```text
RULER:
  keep_recent        score 0.000, collapse 1.000
  keep_early         score 0.333, collapse 0.000
  keep_even          score 0.333, collapse 0.000
  oracle_keep_answer score 0.333, collapse 0.667

SCBench:
  keep_recent        score 0.008, collapse 0.667
  keep_early         score 0.005, collapse 0.000
  keep_even          score 0.004, collapse 0.000
  oracle_keep_answer score 0.007, collapse 0.667
```

## Interpretation

The original `keep_recent` policy is a major source of compressed-generation
collapse on this small subset. Replacing it with `keep_early` or `keep_even`
removes collapse in both RULER and SCBench for this run.

However, `oracle_keep_answer` still collapses on 2/3 RULER and 2/3 SCBench
examples even though it keeps the gold answer whenever possible. This means
that answer retention alone is not sufficient; chunk selection order, chunk
positioning, or the compressed cross-attention path can still destabilize
generation.

The ablation changes the working hypothesis:

```text
Old hypothesis:
  compressed evaluation fails mainly because too much information is dropped.

Updated hypothesis:
  keep_recent is harmful, but generation stability also depends on which
  retained chunks are supplied and how their positions are represented.
```

## Recommended Next Step

Run a slightly larger policy check with only the stable candidates:

```text
benchmarks: RULER, SCBench
compression_ratios: 0.5, 0.25
retention_policies: keep_early, keep_even
max_new_tokens: 16 or 32
```

Do not scale to all benchmarks until `keep_early/keep_even` are verified beyond
this six-example ablation.
