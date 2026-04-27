# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Policy | Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| keep_early | 0.25 | RULER | 12 | 0.083 | 23.51 | 1.573 | 0 |
| keep_early | 0.25 | SCBench | 4 | 0.057 | 26.44 | 1.934 | 0 |
| keep_early | 0.5 | RULER | 12 | 0.083 | 24.71 | 1.664 | 0 |
| keep_early | 0.5 | SCBench | 4 | 0.064 | 30.44 | 2.289 | 0 |
| keep_even | 0.25 | RULER | 12 | 0.167 | 23.41 | 1.497 | 0 |
| keep_even | 0.25 | SCBench | 4 | 0.064 | 26.33 | 1.926 | 0 |
| keep_even | 0.5 | RULER | 12 | 0.083 | 24.61 | 1.667 | 0 |
| keep_even | 0.5 | SCBench | 4 | 0.064 | 30.33 | 2.286 | 0 |

## Task Summary

| Policy | Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | --- | ---: | ---: |
| keep_early | 0.25 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_early | 0.25 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| keep_early | 0.25 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_early | 0.25 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_early | 0.25 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_early | 0.25 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.016 |
| keep_early | 0.25 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.211 |
| keep_early | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_early | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| keep_early | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_early | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_early | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_early | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.016 |
| keep_early | 0.5 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.241 |
| keep_even | 0.25 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_even | 0.25 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.250 |
| keep_even | 0.25 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_even | 0.25 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_even | 0.25 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_even | 0.25 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.016 |
| keep_even | 0.25 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.241 |
| keep_even | 0.5 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.250 |
| keep_even | 0.5 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| keep_even | 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_kv | exact_or_contains | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_qa_eng | qa_f1 | 1 | 0.000 |
| keep_even | 0.5 | SCBench | scbench_repoqa | edit_similarity | 1 | 0.013 |
| keep_even | 0.5 | SCBench | scbench_summary | rouge_l_f1 | 1 | 0.241 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
