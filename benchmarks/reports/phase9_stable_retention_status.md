# Phase 9 Stable Retention Validation Status

## Scope

This phase validates the two stable retention candidates from Phase 8:

```text
keep_early
keep_even
```

It still uses only RULER + SCBench, but expands from the Phase 8 six-example
ablation to the full current dev subset for those two benchmarks.

Run scope:

```text
benchmarks: RULER, SCBench
examples: RULER 12 + SCBench 4 = 16
compression_ratios: 0.5, 0.25
retention_policies: keep_early, keep_even
max_new_tokens: 16
gpu: CUDA_VISIBLE_DEVICES=0
total rows: 64
errors: 0
```

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,SCBench \
  --compression-ratios 0.5,0.25 \
  --retention-policies keep_early,keep_even \
  --max-new-tokens 16 \
  --output-dir benchmarks/results/phase9_stable_retention_ruler_scbench
```

## Outputs

```text
benchmarks/results/phase9_stable_retention_ruler_scbench/
benchmarks/reports/phase9_stable_retention/scored/
benchmarks/reports/phase9_stable_retention/failure_analysis/
```

## Collapse Diagnostics

Collapse is eliminated on this expanded RULER+SCBench dev check:

```text
keep_early, ratio 0.5:
  RULER collapse 0.000
  SCBench collapse 0.000

keep_early, ratio 0.25:
  RULER collapse 0.000
  SCBench collapse 0.000

keep_even, ratio 0.5:
  RULER collapse 0.000
  SCBench collapse 0.000

keep_even, ratio 0.25:
  RULER collapse 0.000
  SCBench collapse 0.000
```

This confirms the Phase 8 finding: `keep_recent` was a major cause of repetitive
compressed-generation collapse.

## Quality Scores

Task-aware diagnostic scores remain low:

```text
RULER:
  keep_early ratio 0.5:  0.083
  keep_early ratio 0.25: 0.083
  keep_even  ratio 0.5:  0.083
  keep_even  ratio 0.25: 0.167

SCBench:
  keep_early ratio 0.5:  0.064
  keep_early ratio 0.25: 0.057
  keep_even  ratio 0.5:  0.064
  keep_even  ratio 0.25: 0.064
```

The model often produces fluent but incorrect or incomplete answers. So Phase 9
solves collapse, not benchmark quality.

## Answer Coverage

Even with no collapse, retained context often does not include the gold answer:

```text
RULER answer dropped:
  keep_early ratio 0.5:  0.500
  keep_early ratio 0.25: 0.500
  keep_even  ratio 0.5:  0.583
  keep_even  ratio 0.25: 0.667

SCBench answer dropped:
  keep_early ratio 0.5:  0.250
  keep_early ratio 0.25: 0.250
  keep_even  ratio 0.5:  0.250
  keep_even  ratio 0.25: 0.000
```

This explains why quality remains low even after generation stabilizes.

## Interpretation

The working conclusion is now:

```text
1. keep_recent should not be the default compressed-context policy for these
   benchmarks; it causes repetitive collapse.
2. keep_early and keep_even are stable enough for further small-scale testing.
3. Compression quality is still weak because these policies often drop the
   evidence needed to answer RULER/SCBench questions.
```

For final benchmark comparison, the compressed CombLlama path still needs a
better evidence-retention strategy or a true KV/cache compressor. But the
evaluation harness can now avoid measuring pure generation collapse.

## Recommended Next Step

Run a broader but still controlled check:

```text
benchmarks: RULER, SCBench, LongBench
retention_policy: keep_even only
compression_ratios: 1.0, 0.5, 0.25
max_new_tokens: 32
```

Rationale:

```text
keep_even avoids collapse, covers positions better than keep_early, and is fair
because it does not use gold answer location.
```

Do not use `oracle_keep_answer` outside diagnostics.
