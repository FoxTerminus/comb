# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Policy | Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| all_encoder_chunks | 1.0 | LoCoMo | 1 | 0.000 | 24.88 | 1.643 | 0 |
| all_encoder_chunks | 1.0 | LongBench | 5 | 0.140 | 23.11 | 3.569 | 0 |
| all_encoder_chunks | 1.0 | LongCodeBench | 3 | 0.000 | 22.48 | 1.426 | 0 |
| all_encoder_chunks | 1.0 | RULER | 12 | 0.167 | 27.01 | 3.439 | 0 |
| all_encoder_chunks | 1.0 | SCBench | 4 | 0.119 | 38.46 | 6.470 | 0 |

## Task Summary

| Policy | Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | --- | ---: | ---: |
| all_encoder_chunks | 1.0 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| all_encoder_chunks | 1.0 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.044 |
| all_encoder_chunks | 1.0 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.118 |
| all_encoder_chunks | 1.0 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| all_encoder_chunks | 1.0 | LongBench | qasper_e | qa_f1 | 1 | 0.356 |
| all_encoder_chunks | 1.0 | LongBench | repobench-p_e | edit_similarity | 1 | 0.092 |
| all_encoder_chunks | 1.0 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| all_encoder_chunks | 1.0 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| all_encoder_chunks | 1.0 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| all_encoder_chunks | 1.0 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| all_encoder_chunks | 1.0 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.250 |
| all_encoder_chunks | 1.0 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| all_encoder_chunks | 1.0 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| all_encoder_chunks | 1.0 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| all_encoder_chunks | 1.0 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.083 |
| all_encoder_chunks | 1.0 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.394 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
