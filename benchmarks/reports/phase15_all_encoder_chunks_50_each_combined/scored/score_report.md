# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Policy | Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| all_encoder_chunks | 1.0 | LoCoMo | 50 | 0.160 | 24.90 | 3.844 | 0 |
| all_encoder_chunks | 1.0 | LongBench | 50 | 0.137 | 23.05 | 3.139 | 0 |
| all_encoder_chunks | 1.0 | LongCodeBench | 50 | 0.700 | 25.76 | 3.992 | 0 |
| all_encoder_chunks | 1.0 | RULER | 50 | 0.040 | 27.60 | 3.929 | 0 |
| all_encoder_chunks | 1.0 | SCBench | 50 | 0.000 | 38.51 | 9.349 | 0 |

## Task Summary

| Policy | Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | --- | ---: | ---: |
| all_encoder_chunks | 1.0 | LoCoMo | multi_hop | choice_match | 24 | 0.083 |
| all_encoder_chunks | 1.0 | LoCoMo | single_hop | choice_match | 19 | 0.316 |
| all_encoder_chunks | 1.0 | LoCoMo | temporal_reasoning | choice_match | 7 | 0.000 |
| all_encoder_chunks | 1.0 | LongBench | qasper_e | qa_f1 | 50 | 0.137 |
| all_encoder_chunks | 1.0 | LongCodeBench | LongCodeQA_32K | choice_match | 50 | 0.700 |
| all_encoder_chunks | 1.0 | RULER | passkey_retrieval_early | passkey_contains | 17 | 0.059 |
| all_encoder_chunks | 1.0 | RULER | passkey_retrieval_late | passkey_contains | 16 | 0.062 |
| all_encoder_chunks | 1.0 | RULER | passkey_retrieval_middle | passkey_contains | 17 | 0.000 |
| all_encoder_chunks | 1.0 | SCBench | scbench_kv | exact_or_contains | 50 | 0.000 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
