# CombLlama Phase 4 Dev Sweep Analysis

## Scope

This report analyzes the CombLlama-only Phase 3 dev sweep. YOCO is excluded until its training finishes.

The sweep covers RULER, LongBench, SCBench, LongCodeBench, and LoCoMo with compression ratios `1.0`, `0.5`, `0.25`, and `0.125`.

Generation length is `max_new_tokens=8`, so the quality metrics are diagnostic rather than final official benchmark scores.

## Overall By Compression Ratio

| Compression | Examples | Contains | F1 | Avg Chunk Tokens | Max Mem GB | Prefill s | Decode s | Mem Saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 25 | 0.000 | 0.032 | 23933.1 | 38.47 | 0.570 | 3.902 | 0.0% |
| 0.5 | 25 | 0.000 | 0.000 | 11563.2 | 30.33 | 0.320 | 2.247 | 21.1% |
| 0.25 | 25 | 0.000 | 0.000 | 5542.0 | 26.33 | 0.205 | 1.439 | 31.5% |
| 0.125 | 25 | 0.000 | 0.000 | 2715.8 | 24.33 | 0.150 | 1.057 | 36.7% |

## By Benchmark

| Compression | Benchmark | Examples | Contains | F1 | Max Mem GB | Decode s | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | LoCoMo | 1 | 0.000 | 0.000 | 22.62 | 0.921 | 0 |
| 0.125 | LongBench | 5 | 0.000 | 0.000 | 22.62 | 0.865 | 0 |
| 0.125 | LongCodeBench | 3 | 0.000 | 0.000 | 22.47 | 0.205 | 0 |
| 0.125 | RULER | 12 | 0.000 | 0.000 | 22.88 | 0.975 | 0 |
| 0.125 | SCBench | 4 | 0.000 | 0.000 | 24.33 | 2.218 | 0 |
| 0.25 | LoCoMo | 1 | 0.000 | 0.000 | 22.88 | 1.193 | 0 |
| 0.25 | LongBench | 5 | 0.000 | 0.000 | 22.62 | 0.864 | 0 |
| 0.25 | LongCodeBench | 3 | 0.000 | 0.000 | 22.47 | 0.205 | 0 |
| 0.25 | RULER | 12 | 0.000 | 0.000 | 23.42 | 1.224 | 0 |
| 0.25 | SCBench | 4 | 0.000 | 0.000 | 26.33 | 3.792 | 0 |
| 0.5 | LoCoMo | 1 | 0.000 | 0.000 | 23.54 | 1.886 | 0 |
| 0.5 | LongBench | 5 | 0.000 | 0.000 | 22.71 | 0.948 | 0 |
| 0.5 | LongCodeBench | 3 | 0.000 | 0.000 | 22.47 | 0.205 | 0 |
| 0.5 | RULER | 12 | 0.000 | 0.000 | 24.62 | 1.773 | 0 |
| 0.5 | SCBench | 4 | 0.000 | 0.000 | 30.33 | 6.911 | 0 |
| 1.0 | LoCoMo | 1 | 0.000 | 0.444 | 24.88 | 3.263 | 0 |
| 1.0 | LongBench | 5 | 0.000 | 0.015 | 23.11 | 1.195 | 0 |
| 1.0 | LongCodeBench | 3 | 0.000 | 0.000 | 22.47 | 0.206 | 0 |
| 1.0 | RULER | 12 | 0.000 | 0.000 | 27.02 | 2.901 | 0 |
| 1.0 | SCBench | 4 | 0.000 | 0.068 | 38.47 | 13.219 | 0 |

## Interpretation

The efficiency trend is clear: retaining fewer historical chunks reduces memory and latency monotonically.

Quality degrades sharply when the retained compressed context is reduced. This is visible in both the generated text and the near-zero contains/F1 metrics at lower ratios.

The current low exact-match scores should not be treated as final model quality because answers are often longer than 8 tokens and official task metrics have not yet been plugged in.

## Next Engineering Step

Implement and verify self-attention KV cache reuse in CombLlama inference. Without this, long-answer evaluation with `max_new_tokens=32+` is too slow because each generated token re-runs the packed context.
