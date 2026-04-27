# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Policy | Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| keep_even | 0.25 | LongBench | 5 | 0.066 | 22.62 | 3.281 | 0 |
| keep_even | 0.25 | RULER | 12 | 0.167 | 23.41 | 2.465 | 0 |
| keep_even | 0.25 | SCBench | 4 | 0.074 | 26.33 | 3.980 | 0 |
| keep_even | 0.5 | LongBench | 5 | 0.066 | 22.71 | 3.298 | 0 |
| keep_even | 0.5 | RULER | 12 | 0.083 | 24.62 | 3.006 | 0 |
| keep_even | 0.5 | SCBench | 4 | 0.136 | 30.33 | 4.734 | 0 |
| keep_even | 1.0 | LongBench | 5 | 0.140 | 23.11 | 3.347 | 0 |
| keep_even | 1.0 | RULER | 12 | 0.167 | 27.01 | 3.290 | 0 |
| keep_even | 1.0 | SCBench | 4 | 0.119 | 38.46 | 6.258 | 0 |

## Task Summary

| Policy | Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | --- | ---: | ---: |
| keep_even | 0.25 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.031 |
| keep_even | 0.25 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.118 |
| keep_even | 0.25 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| keep_even | 0.25 | LongBench | qasper_e | qa_f1 | 1 | 0.000 |
| keep_even | 0.25 | LongBench | repobench-p_e | edit_similarity | 1 | 0.092 |
| keep_even | 0.25 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_even | 0.25 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.250 |
| keep_even | 0.25 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_even | 0.25 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_even | 0.25 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_even | 0.25 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.071 |
| keep_even | 0.25 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.225 |
| keep_even | 0.5 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.031 |
| keep_even | 0.5 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.118 |
| keep_even | 0.5 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| keep_even | 0.5 | LongBench | qasper_e | qa_f1 | 1 | 0.000 |
| keep_even | 0.5 | LongBench | repobench-p_e | edit_similarity | 1 | 0.092 |
| keep_even | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_even | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| keep_even | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.151 |
| keep_even | 0.5 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.394 |
| keep_even | 1.0 | LongBench | gov_report_e | rouge_l_f1 | 1 | 0.044 |
| keep_even | 1.0 | LongBench | hotpotqa_e | qa_f1 | 1 | 0.118 |
| keep_even | 1.0 | LongBench | lcc_e | edit_similarity | 1 | 0.090 |
| keep_even | 1.0 | LongBench | qasper_e | qa_f1 | 1 | 0.356 |
| keep_even | 1.0 | LongBench | repobench-p_e | edit_similarity | 1 | 0.092 |
| keep_even | 1.0 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_even | 1.0 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.250 |
| keep_even | 1.0 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_even | 1.0 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_even | 1.0 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_even | 1.0 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.083 |
| keep_even | 1.0 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.394 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
