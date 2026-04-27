# Phase 12: CombLlama Interim Benchmark Report

## Scope

This is a CombLlama-only interim report. YOCO is still training, so this report does not make YOCO-vs-CombLlama claims.

Current evaluated model:

- Model: `/data3/junhaohu/model/CombLlama-8B-Instruct`
- Tokenizer: local Llama-3.1-8B-Instruct snapshot
- Precision: `bfloat16`
- Generation: greedy, `max_new_tokens=32`
- Chunk encoder cache: enabled
- Self-attention KV cache: not used

Current core dev subset:

- RULER: 12 passkey retrieval samples.
- SCBench: 4 shared-context samples.
- LongBench: 5 QA/summary/code-context samples.

LongCodeBench and LoCoMo are present in the benchmark tree, but they are not used for the main Phase 10-11 decision because their current dev subsets are too small and under-calibrated for stable policy selection.

## Current Default Policy

Use:

```json
"retention_policy": "keep_even"
```

Do not use `keep_recent` as the default compressed-context policy. Earlier phases showed it can cause severe degeneration when compression drops earlier evidence.

Do not use `keep_early_recent_even` as the default policy. Phase 11 showed that it reintroduces collapse on LongBench.

## Quality and Efficiency: keep_even

The most reliable current result is Phase 10, using `keep_even` on RULER, SCBench, and LongBench.

| Compression | Benchmark | Examples | Score | Max Memory GB | Prefill s | Decode s | Errors | Collapse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | RULER | 12 | 0.167 | 27.01 | 0.437 | 3.290 | 0 | 0.000 |
| 1.0 | SCBench | 4 | 0.119 | 38.46 | 1.883 | 6.258 | 0 | 0.000 |
| 1.0 | LongBench | 5 | 0.140 | 23.11 | 0.171 | 3.347 | 0 | 0.000 |
| 0.5 | RULER | 12 | 0.083 | 24.62 | 0.254 | 3.006 | 0 | 0.000 |
| 0.5 | SCBench | 4 | 0.136 | 30.33 | 0.986 | 4.734 | 0 | 0.000 |
| 0.5 | LongBench | 5 | 0.066 | 22.71 | 0.136 | 3.298 | 0 | 0.000 |
| 0.25 | RULER | 12 | 0.167 | 23.41 | 0.178 | 2.465 | 0 | 0.000 |
| 0.25 | SCBench | 4 | 0.074 | 26.33 | 0.541 | 3.980 | 0 | 0.000 |
| 0.25 | LongBench | 5 | 0.066 | 22.62 | 0.128 | 3.281 | 0 | 0.000 |

Interpretation:

- `keep_even` is stable: no generation errors and no collapse on the tested core set.
- Compression gives a clear memory benefit, especially on SCBench: max memory drops from `38.46 GB` at full context to `30.33 GB` at 0.5 and `26.33 GB` at 0.25.
- Quality remains weak on the small dev subset. This is not only a compression issue; even full-context scores are low on some tasks.

## Retention Policy Ablation

Phase 11 compared `keep_even` with `keep_early_recent_even`.

| Policy | Compression | Benchmark | Score | Collapse |
| --- | --- | --- | ---: | ---: |
| keep_even | 0.5 | RULER | 0.083 | 0.000 |
| keep_even | 0.5 | SCBench | 0.136 | 0.000 |
| keep_even | 0.5 | LongBench | 0.066 | 0.000 |
| keep_even | 0.25 | RULER | 0.167 | 0.000 |
| keep_even | 0.25 | SCBench | 0.074 | 0.000 |
| keep_even | 0.25 | LongBench | 0.066 | 0.000 |
| keep_early_recent_even | 0.5 | RULER | 0.083 | 0.000 |
| keep_early_recent_even | 0.5 | SCBench | 0.136 | 0.000 |
| keep_early_recent_even | 0.5 | LongBench | 0.060 | 0.200 |
| keep_early_recent_even | 0.25 | RULER | 0.083 | 0.167 |
| keep_early_recent_even | 0.25 | SCBench | 0.068 | 0.000 |
| keep_early_recent_even | 0.25 | LongBench | 0.028 | 0.800 |

Decision:

- Keep `keep_even` as the default fair compression policy.
- Keep `keep_early_recent_even` only as an experimental diagnostic policy.
- Do not add more retention heuristics before broadening the evaluation set.

## Failure Modes

The current failures split into two categories.

Evidence dropping:

- RULER answer-in-dropped rate remains high under compression.
- At 0.5, RULER answer-in-dropped rate is `0.583`.
- At 0.25, RULER answer-in-dropped rate is `0.667`.
- This explains why stable generation does not necessarily improve retrieval quality.

Generation collapse:

- `keep_even` avoids collapse on the tested Phase 10 core set.
- `keep_early_recent_even` reintroduces collapse on LongBench, especially at 0.25.
- The collapse issue is therefore policy-dependent, not simply caused by model loading or GPU/runtime instability.

## What This Report Can Support

Current supported claims:

- The benchmark harness can run CombLlama deterministically on local dev subsets.
- Chunk-state caching makes `max_new_tokens=32` dev sweeps practical.
- `keep_even` is the current safest fair compressed-context policy.
- Compression reduces memory and prefill cost.
- The dominant quality bottleneck is evidence retention, not runtime failure.

Current unsupported claims:

- No final YOCO-vs-CombLlama comparison yet.
- No official benchmark leaderboard scores yet.
- No statistically strong conclusion from LongCodeBench or LoCoMo yet.
- No claim about production decoding efficiency with self-attention KV cache, because the adapter still re-runs the recent decoder window.

## Recommended Next Steps

1. Keep `keep_even` as the default while YOCO training finishes.
2. Broaden the dev subsets before adding more compression heuristics.
3. Add official evaluator integration for LongBench, SCBench, RULER, LongCodeBench, and LoCoMo before final reporting.
4. Once YOCO is ready, run the same benchmark matrix with matched context length, generation settings, and sample splits.
5. Produce the final comparison with quality, max memory, prefill latency, decode latency, and failure categories.

## Key Artifacts

- Phase 10 status: `benchmarks/reports/phase10_keep_even_status.md`
- Phase 10 scores: `benchmarks/reports/phase10_keep_even/scored/benchmark_summary.csv`
- Phase 10 diagnostics: `benchmarks/reports/phase10_keep_even/failure_analysis/collapse_summary.csv`
- Phase 11 status: `benchmarks/reports/phase11_retention_policy_comparison_status.md`
- Phase 11 scores: `benchmarks/reports/phase11_retention_policy_comparison/scored/benchmark_summary.csv`
- Phase 11 diagnostics: `benchmarks/reports/phase11_retention_policy_comparison/failure_analysis/collapse_summary.csv`
