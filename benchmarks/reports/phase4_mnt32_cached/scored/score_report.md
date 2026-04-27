# Scored Benchmark Diagnostics

These scores are task-aware diagnostics for the current dev subset. They are closer to benchmark semantics than raw exact match, but still not a replacement for each benchmark's official evaluator.

## Benchmark Summary

| Compression | Benchmark | Examples | Score | Max Mem GB | Decode s | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.125 | LoCoMo | 1 | 0.000 | 22.62 | 3.256 | 0 |
| 0.125 | LongBench | 5 | 0.000 | 22.62 | 3.247 | 0 |
| 0.125 | LongCodeBench | 3 | 0.000 | 22.47 | 0.602 | 0 |
| 0.125 | RULER | 12 | 0.000 | 22.88 | 3.266 | 0 |
| 0.125 | SCBench | 4 | 0.000 | 24.33 | 3.576 | 0 |
| 0.25 | LoCoMo | 1 | 0.000 | 22.88 | 3.315 | 0 |
| 0.25 | LongBench | 5 | 0.000 | 22.62 | 3.246 | 0 |
| 0.25 | LongCodeBench | 3 | 0.000 | 22.47 | 0.601 | 0 |
| 0.25 | RULER | 12 | 0.000 | 23.42 | 3.326 | 0 |
| 0.25 | SCBench | 4 | 0.000 | 26.33 | 3.948 | 0 |
| 0.5 | LoCoMo | 1 | 0.000 | 23.54 | 3.493 | 0 |
| 0.5 | LongBench | 5 | 0.000 | 22.71 | 3.259 | 0 |
| 0.5 | LongCodeBench | 3 | 0.000 | 22.47 | 0.603 | 0 |
| 0.5 | RULER | 12 | 0.000 | 24.62 | 3.443 | 0 |
| 0.5 | SCBench | 4 | 0.000 | 30.33 | 4.701 | 0 |
| 1.0 | LoCoMo | 1 | 1.000 | 24.88 | 1.365 | 0 |
| 1.0 | LongBench | 5 | 0.200 | 23.11 | 3.348 | 0 |
| 1.0 | LongCodeBench | 3 | 0.000 | 22.47 | 0.604 | 0 |
| 1.0 | RULER | 12 | 0.083 | 27.01 | 3.221 | 0 |
| 1.0 | SCBench | 4 | 0.000 | 38.46 | 6.260 | 0 |

## Task Summary

| Compression | Benchmark | Task | Metric | Examples | Score |
| --- | --- | --- | --- | ---: | ---: |
| 0.125 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.125 | LongBench | gov_report_e | contains_match | 1 | 0.000 |
| 0.125 | LongBench | hotpotqa_e | contains_match | 1 | 0.000 |
| 0.125 | LongBench | lcc_e | contains_match | 1 | 0.000 |
| 0.125 | LongBench | qasper_e | contains_match | 1 | 0.000 |
| 0.125 | LongBench | repobench-p_e | contains_match | 1 | 0.000 |
| 0.125 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 0.125 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 0.125 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 0.125 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 0.125 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| 0.125 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 0.125 | SCBench | scbench_kv | contains_match | 1 | 0.000 |
| 0.125 | SCBench | scbench_qa_eng | contains_match | 1 | 0.000 |
| 0.125 | SCBench | scbench_repoqa | contains_match | 1 | 0.000 |
| 0.125 | SCBench | scbench_summary | contains_match | 1 | 0.000 |
| 0.25 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.25 | LongBench | gov_report_e | contains_match | 1 | 0.000 |
| 0.25 | LongBench | hotpotqa_e | contains_match | 1 | 0.000 |
| 0.25 | LongBench | lcc_e | contains_match | 1 | 0.000 |
| 0.25 | LongBench | qasper_e | contains_match | 1 | 0.000 |
| 0.25 | LongBench | repobench-p_e | contains_match | 1 | 0.000 |
| 0.25 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 0.25 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 0.25 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 0.25 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 0.25 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| 0.25 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 0.25 | SCBench | scbench_kv | contains_match | 1 | 0.000 |
| 0.25 | SCBench | scbench_qa_eng | contains_match | 1 | 0.000 |
| 0.25 | SCBench | scbench_repoqa | contains_match | 1 | 0.000 |
| 0.25 | SCBench | scbench_summary | contains_match | 1 | 0.000 |
| 0.5 | LoCoMo | multi_hop | choice_match | 1 | 0.000 |
| 0.5 | LongBench | gov_report_e | contains_match | 1 | 0.000 |
| 0.5 | LongBench | hotpotqa_e | contains_match | 1 | 0.000 |
| 0.5 | LongBench | lcc_e | contains_match | 1 | 0.000 |
| 0.5 | LongBench | qasper_e | contains_match | 1 | 0.000 |
| 0.5 | LongBench | repobench-p_e | contains_match | 1 | 0.000 |
| 0.5 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 0.5 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 0.5 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 0.5 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 0.5 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.000 |
| 0.5 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 0.5 | SCBench | scbench_kv | contains_match | 1 | 0.000 |
| 0.5 | SCBench | scbench_qa_eng | contains_match | 1 | 0.000 |
| 0.5 | SCBench | scbench_repoqa | contains_match | 1 | 0.000 |
| 0.5 | SCBench | scbench_summary | contains_match | 1 | 0.000 |
| 1.0 | LoCoMo | multi_hop | choice_match | 1 | 1.000 |
| 1.0 | LongBench | gov_report_e | contains_match | 1 | 0.000 |
| 1.0 | LongBench | hotpotqa_e | contains_match | 1 | 1.000 |
| 1.0 | LongBench | lcc_e | contains_match | 1 | 0.000 |
| 1.0 | LongBench | qasper_e | contains_match | 1 | 0.000 |
| 1.0 | LongBench | repobench-p_e | contains_match | 1 | 0.000 |
| 1.0 | LongCodeBench | LongCodeQA_128K | choice_match | 1 | 0.000 |
| 1.0 | LongCodeBench | LongCodeQA_32K | choice_match | 1 | 0.000 |
| 1.0 | LongCodeBench | LongCodeQA_64K | choice_match | 1 | 0.000 |
| 1.0 | RULER | passkey_retrieval_early | passkey_contains | 4 | 0.000 |
| 1.0 | RULER | passkey_retrieval_late | passkey_contains | 4 | 0.250 |
| 1.0 | RULER | passkey_retrieval_middle | passkey_contains | 4 | 0.000 |
| 1.0 | SCBench | scbench_kv | contains_match | 1 | 0.000 |
| 1.0 | SCBench | scbench_qa_eng | contains_match | 1 | 0.000 |
| 1.0 | SCBench | scbench_repoqa | contains_match | 1 | 0.000 |
| 1.0 | SCBench | scbench_summary | contains_match | 1 | 0.000 |

## Notes

RULER uses passkey containment.

LongCodeBench and LoCoMo use multiple-choice style matching when choices are available.

SCBench and LongBench use contains-match fallback until official evaluators are wired in.
