# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. LongBench and SCBench now use local near-official metric families, but full paper/report numbers should still be regenerated with official evaluators before final claims.

## Benchmark Summary

| Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.125 | LoCoMo | 1 | 0.000 | 22.63 | 3.236 | 0 |
| 0.25 | LoCoMo | 1 | 0.000 | 22.88 | 3.296 | 0 |
| 0.5 | LoCoMo | 1 | 0.000 | 23.54 | 3.464 | 0 |
| 1.0 | LoCoMo | 1 | 0.000 | 24.88 | 3.812 | 0 |

## Task Summary

| Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | ---: | ---: |
| 0.125 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.25 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.5 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 1.0 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

LongBench uses QA F1, Rouge-L F1, exact match, or edit similarity according to task family.

SCBench uses Rouge-L F1 for summaries, edit similarity for repo/code QA, exact-or-contains for KV retrieval, and QA F1 for open QA.
