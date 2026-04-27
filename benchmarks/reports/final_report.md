# CombLlama Benchmark Report

## Scope

This is the current consolidated benchmark report for the local CombLlama harness. It supersedes the older phase-by-phase reports, which have been removed to keep only the latest result set.

Current result semantics:

- Model: `/data3/junhaohu/model/CombLlama-8B-Instruct`
- Policy: `all_encoder_chunks`
- Compression ratio: `1.0`
- Generation: greedy, `max_new_tokens=32`
- Split: local deterministic dev subsets
- Samples: 50 examples per benchmark, 250 examples total

Under `all_encoder_chunks`, every history chunk is sent through the CombLlama chunk encoder. The benchmark harness does not drop evidence chunks. Older `keep_*` retention runs are retained only as historical ablations and should not be used as the main CombLlama result.

## Current Result

| Benchmark | Examples | Score | Avg Memory GB | Max Memory GB | Prefill s | Decode s | Errors | Collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 50 | 0.160 | 24.88 | 24.90 | 0.478 | 3.844 | 0 | 0.000 |
| LongBench | 50 | 0.137 | 22.85 | 23.05 | 0.177 | 3.139 | 0 | 0.000 |
| LongCodeBench | 50 | 0.700 | 24.90 | 25.76 | 0.486 | 3.992 | 0 | 0.000 |
| RULER | 50 | 0.040 | 25.21 | 27.60 | 0.547 | 3.929 | 0 | 0.000 |
| SCBench | 50 | 0.000 | 38.47 | 38.51 | 3.048 | 9.349 | 0 | 0.000 |

## Task Detail

| Benchmark | Task | Examples | Metric | Score |
| --- | --- | ---: | --- | ---: |
| LoCoMo | multi_hop | 24 | choice_match | 0.083 |
| LoCoMo | single_hop | 19 | choice_match | 0.316 |
| LoCoMo | temporal_reasoning | 7 | choice_match | 0.000 |
| LongBench | qasper_e | 50 | qa_f1 | 0.137 |
| LongCodeBench | LongCodeQA_32K | 50 | choice_match | 0.700 |
| RULER | passkey_retrieval_early | 17 | passkey_contains | 0.059 |
| RULER | passkey_retrieval_middle | 17 | passkey_contains | 0.000 |
| RULER | passkey_retrieval_late | 16 | passkey_contains | 0.062 |
| SCBench | scbench_kv | 50 | exact_or_contains | 0.000 |

## Interpretation

The harness is now stable for a 250-example CombLlama run with no runtime errors, no benchmark-side chunk dropping, and no detected repetitive generation collapse.

Quality is uneven. LongCodeBench is currently strongest on the selected local subset. LoCoMo has limited success, mainly on single-hop questions. LongBench Qasper remains low. RULER passkey retrieval is weak, especially middle-position retrieval. SCBench KV retrieval is currently failing under the local metric despite successful generation.

These are local diagnostic scores, not official benchmark leaderboard scores. The subsets are deterministic but not fully balanced: LongBench is all `qasper_e`, LongCodeBench is all `LongCodeQA_32K`, and SCBench is all `scbench_kv`.

## Files To Use

- Main report: `benchmarks/reports/final_report.md`
- Benchmark summary: `benchmarks/reports/current_benchmark_summary.csv`
- Task summary: `benchmarks/reports/current_task_summary.csv`
- Failure summary: `benchmarks/reports/current_failure_summary.csv`
- Raw predictions: `benchmarks/results/current/dev_predictions_all_encoder_chunks_cr1p0.jsonl`
- Manifest: `benchmarks/results/current/dev_sweep_manifest.json`

## Historical Boundary

Older phase outputs were removed. The important historical boundary is Phase 13/14: those phases corrected the benchmark semantics from chunk-dropping retention ablations to the current `all_encoder_chunks` path.
