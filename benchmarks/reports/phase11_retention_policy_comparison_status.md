# Phase 11: Retention Policy Comparison

## Scope

This phase tests a new fair, answer-agnostic policy, `keep_early_recent_even`, against the current stable policy, `keep_even`.

The comparison uses CombLlama on the current dev subsets:

- RULER: 12 passkey retrieval examples.
- SCBench: 4 long-context KV/code/QA/summary examples.
- LongBench: 5 QA/summary/code-context examples.

Compression ratios: `0.5` and `0.25`.

## Code Change

`benchmarks/scripts/prompting.py` now supports:

- `keep_early_recent_even`: reserves chunks from the beginning and end of the history, then fills the remaining budget with evenly distributed middle chunks.

The default `retention_policy` in `benchmarks/configs/combllama_phase3_sweep.json` is now `keep_even`, because earlier phases showed that `keep_recent` is unstable under compression.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,SCBench,LongBench \
  --compression-ratios 0.5,0.25 \
  --retention-policies keep_even,keep_early_recent_even \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase11_retention_policy_comparison
```

Scoring and diagnostics were generated under:

- `benchmarks/reports/phase11_retention_policy_comparison/scored/`
- `benchmarks/reports/phase11_retention_policy_comparison/failure_analysis/`

## Results

| Policy | Compression | Benchmark | Score | Errors | Collapse Rate |
| --- | --- | --- | ---: | ---: | ---: |
| keep_even | 0.5 | RULER | 0.083 | 0 | 0.000 |
| keep_even | 0.5 | SCBench | 0.136 | 0 | 0.000 |
| keep_even | 0.5 | LongBench | 0.066 | 0 | 0.000 |
| keep_even | 0.25 | RULER | 0.167 | 0 | 0.000 |
| keep_even | 0.25 | SCBench | 0.074 | 0 | 0.000 |
| keep_even | 0.25 | LongBench | 0.066 | 0 | 0.000 |
| keep_early_recent_even | 0.5 | RULER | 0.083 | 0 | 0.000 |
| keep_early_recent_even | 0.5 | SCBench | 0.136 | 0 | 0.000 |
| keep_early_recent_even | 0.5 | LongBench | 0.060 | 0 | 0.200 |
| keep_early_recent_even | 0.25 | RULER | 0.083 | 0 | 0.167 |
| keep_early_recent_even | 0.25 | SCBench | 0.068 | 0 | 0.000 |
| keep_early_recent_even | 0.25 | LongBench | 0.028 | 0 | 0.800 |

## Interpretation

`keep_early_recent_even` should not replace `keep_even`. It does not improve benchmark score on the tested dev subset, and it reintroduces generation collapse on LongBench, especially at compression ratio `0.25`.

The likely reason is that preserving both boundary regions while shrinking the middle budget changes the chunk distribution enough to destabilize some long-document generations. It also does not materially improve answer retention: RULER answer-in-dropped rate is worse at `0.25` (`0.75` vs `0.667` for `keep_even`).

`keep_even` remains the current default fair compressed-context policy for CombLlama benchmark runs.

## Next Step

Phase 12 should avoid adding more heuristic retention policies until the evaluation set is broadened. The more useful next step is to produce a concise CombLlama-only interim report from Phases 10-11, with:

- Core quality table by benchmark and compression ratio.
- Memory/latency tradeoff table.
- Failure-mode summary: evidence dropping vs generation collapse.
- Clear recommendation that `keep_even` is the only currently stable fair policy.
