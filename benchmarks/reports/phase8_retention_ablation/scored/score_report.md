# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Policy | Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| keep_early | 0.5 | RULER | 3 | 0.333 | 22.71 | 1.559 | 0 |
| keep_early | 0.5 | SCBench | 3 | 0.005 | 30.44 | 2.277 | 0 |
| keep_even | 0.5 | RULER | 3 | 0.333 | 22.64 | 1.553 | 0 |
| keep_even | 0.5 | SCBench | 3 | 0.004 | 30.33 | 2.273 | 0 |
| keep_recent | 0.5 | RULER | 3 | 0.000 | 22.64 | 1.568 | 0 |
| keep_recent | 0.5 | SCBench | 3 | 0.008 | 30.33 | 2.264 | 0 |
| oracle_keep_answer | 0.5 | RULER | 3 | 0.333 | 22.71 | 1.578 | 0 |
| oracle_keep_answer | 0.5 | SCBench | 3 | 0.007 | 30.44 | 2.269 | 0 |

## Task Summary

| Policy | Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | --- | ---: | ---: |
| keep_early | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 1 | 1.000 |
| keep_early | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 1 | 0.000 |
| keep_early | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 1 | 0.000 |
| keep_early | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_early | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_early | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.016 |
| keep_even | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 1 | 1.000 |
| keep_even | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 1 | 0.000 |
| keep_even | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.013 |
| keep_recent | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 1 | 0.000 |
| keep_recent | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 1 | 0.000 |
| keep_recent | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 1 | 0.000 |
| keep_recent | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_recent | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_recent | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.024 |
| oracle_keep_answer | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 1 | 1.000 |
| oracle_keep_answer | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 1 | 0.000 |
| oracle_keep_answer | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 1 | 0.000 |
| oracle_keep_answer | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| oracle_keep_answer | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| oracle_keep_answer | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.020 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
