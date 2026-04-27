# CombLlama Phase 4 Dev Sweep Analysis

## Scope

This report analyzes the CombLlama-only Phase 3 dev sweep. YOCO is excluded until its training finishes.

The sweep covers RULER, LongBench, SCBench, LongCodeBench, and LoCoMo with compression ratios `1.0`, `0.5`, `0.25`, and `0.125`.

This run generated up to `32` tokens per example, with average generated tokens `31.5`.

Quality metrics are still diagnostic because official benchmark-specific scoring has not yet been plugged in.

## Overall By Compression Ratio

| Compression | Examples | Contains | F1 | Avg Chunk Tokens | Max Mem GB | Prefill s | Decode s | Mem Saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 25 | 0.200 | 0.106 | 23927.2 | 38.46 | 0.568 | 3.517 | 0.0% |
| 0.5 | 25 | 0.200 | 0.000 | 11557.2 | 30.33 | 0.320 | 3.309 | 21.1% |
| 0.25 | 25 | 0.160 | 0.000 | 5536.1 | 26.33 | 0.205 | 3.123 | 31.5% |
| 0.125 | 25 | 0.120 | 0.000 | 2709.9 | 24.33 | 0.151 | 3.030 | 36.7% |

## By Benchmark

| Compression | Benchmark | Examples | Contains | F1 | Max Mem GB | Decode s | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | LoCoMo | 1 | 0.000 | 0.000 | 22.62 | 3.249 | 0 |
| 0.125 | LongBench | 5 | 0.000 | 0.000 | 22.62 | 3.247 | 0 |
| 0.125 | LongCodeBench | 3 | 0.667 | 0.000 | 22.48 | 0.938 | 0 |
| 0.125 | RULER | 12 | 0.000 | 0.000 | 22.88 | 3.266 | 0 |
| 0.125 | SCBench | 4 | 0.250 | 0.000 | 24.33 | 3.567 | 0 |
| 0.25 | LoCoMo | 1 | 1.000 | 0.000 | 22.88 | 3.318 | 0 |
| 0.25 | LongBench | 5 | 0.000 | 0.000 | 22.62 | 3.247 | 0 |
| 0.25 | LongCodeBench | 3 | 0.667 | 0.000 | 22.48 | 0.938 | 0 |
| 0.25 | RULER | 12 | 0.000 | 0.000 | 23.41 | 3.326 | 0 |
| 0.25 | SCBench | 4 | 0.250 | 0.000 | 26.33 | 3.948 | 0 |
| 0.5 | LoCoMo | 1 | 1.000 | 0.000 | 23.54 | 3.492 | 0 |
| 0.5 | LongBench | 5 | 0.200 | 0.000 | 22.71 | 3.259 | 0 |
| 0.5 | LongCodeBench | 3 | 0.667 | 0.000 | 22.48 | 0.939 | 0 |
| 0.5 | RULER | 12 | 0.000 | 0.000 | 24.62 | 3.443 | 0 |
| 0.5 | SCBench | 4 | 0.250 | 0.000 | 30.33 | 4.698 | 0 |
| 1.0 | LoCoMo | 1 | 0.000 | 0.190 | 24.88 | 3.848 | 0 |
| 1.0 | LongBench | 5 | 0.200 | 0.106 | 23.11 | 3.344 | 0 |
| 1.0 | LongCodeBench | 3 | 0.667 | 0.000 | 22.48 | 0.940 | 0 |
| 1.0 | RULER | 12 | 0.167 | 0.114 | 27.01 | 3.292 | 0 |
| 1.0 | SCBench | 4 | 0.000 | 0.141 | 38.46 | 6.260 | 0 |

## Interpretation

The efficiency trend is clear: retaining fewer historical chunks reduces memory and latency monotonically.

Quality degrades sharply when the retained compressed context is reduced. This is visible in both the generated text and the near-zero contains/F1 metrics at lower ratios.

The current low exact-match scores should not be treated as final model quality because official task metrics have not yet been plugged in and prompt/task formatting is still preliminary.

## Next Engineering Step

Implement and verify self-attention KV cache reuse in CombLlama inference. Chunk encoder caching now makes `max_new_tokens=32` dev sweeps feasible, but longer generation and larger full benchmark runs still re-run the recent decoder window for each generated token.
