# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.125 | LoCoMo | 1 | 0.000 | 22.62 | 3.249 | 0 |
| 0.125 | LongBench | 5 | 0.028 | 22.62 | 3.247 | 0 |
| 0.125 | LongCodeBench | 3 | 0.000 | 22.48 | 0.938 | 0 |
| 0.125 | RULER | 12 | 0.000 | 22.88 | 3.266 | 0 |
| 0.125 | SCBench | 4 | 0.007 | 24.33 | 3.567 | 0 |
| 0.25 | LoCoMo | 1 | 0.000 | 22.88 | 3.318 | 0 |
| 0.25 | LongBench | 5 | 0.028 | 22.62 | 3.247 | 0 |
| 0.25 | LongCodeBench | 3 | 0.000 | 22.48 | 0.938 | 0 |
| 0.25 | RULER | 12 | 0.000 | 23.41 | 3.326 | 0 |
| 0.25 | SCBench | 4 | 0.004 | 26.33 | 3.948 | 0 |
| 0.5 | LoCoMo | 1 | 0.000 | 23.54 | 3.492 | 0 |
| 0.5 | LongBench | 5 | 0.030 | 22.71 | 3.259 | 0 |
| 0.5 | LongCodeBench | 3 | 0.000 | 22.48 | 0.939 | 0 |
| 0.5 | RULER | 12 | 0.000 | 24.62 | 3.443 | 0 |
| 0.5 | SCBench | 4 | 0.006 | 30.33 | 4.698 | 0 |
| 1.0 | LoCoMo | 1 | 0.000 | 24.88 | 3.848 | 0 |
| 1.0 | LongBench | 5 | 0.140 | 23.11 | 3.344 | 0 |
| 1.0 | LongCodeBench | 3 | 0.000 | 22.48 | 0.940 | 0 |
| 1.0 | RULER | 12 | 0.167 | 27.01 | 3.292 | 0 |
| 1.0 | SCBench | 4 | 0.119 | 38.46 | 6.260 | 0 |

## Task Summary

| Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | ---: | ---: |
| 0.125 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.125 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.000 |
| 0.125 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.000 |
| 0.125 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| 0.125 | LongBench | qasper_e | qa_f1 | 1 | 0.000 |
| 0.125 | LongBench | repobench-p_e | edit_similarity | 1 | 0.051 |
| 0.125 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 0.125 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 0.125 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 0.125 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 0.125 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| 0.125 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 0.125 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| 0.125 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| 0.125 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.030 |
| 0.125 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.000 |
| 0.25 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.25 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.000 |
| 0.25 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.000 |
| 0.25 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| 0.25 | LongBench | qasper_e | qa_f1 | 1 | 0.000 |
| 0.25 | LongBench | repobench-p_e | edit_similarity | 1 | 0.051 |
| 0.25 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 0.25 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 0.25 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 0.25 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 0.25 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| 0.25 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 0.25 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| 0.25 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| 0.25 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.017 |
| 0.25 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.000 |
| 0.5 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.5 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.000 |
| 0.5 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.000 |
| 0.5 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| 0.5 | LongBench | qasper_e | qa_f1 | 1 | 0.000 |
| 0.5 | LongBench | repobench-p_e | edit_similarity | 1 | 0.057 |
| 0.5 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 0.5 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 0.5 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 0.5 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 0.5 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.023 |
| 0.5 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.000 |
| 1.0 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 1.0 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.044 |
| 1.0 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.118 |
| 1.0 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| 1.0 | LongBench | qasper_e | qa_f1 | 1 | 0.356 |
| 1.0 | LongBench | repobench-p_e | edit_similarity | 1 | 0.092 |
| 1.0 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 1.0 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 1.0 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 1.0 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| 1.0 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.250 |
| 1.0 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 1.0 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| 1.0 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| 1.0 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.083 |
| 1.0 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.394 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
