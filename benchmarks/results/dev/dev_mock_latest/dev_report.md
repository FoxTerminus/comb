# Dev Report

Total records: 80

| Group | Count | Success | Failure | Primary | EM | Contains | F1 | Rouge-L | Code/Edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benchmark/LoCoMo | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| benchmark/LongBench | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| benchmark/LongCodeBench | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  | 1.0000 |
| benchmark/RULER | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| benchmark/SCBench | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| kv_cache_policy/mock_no_kv_cache | 80 | 80 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  | 1.0000 |
| length_bucket/<= 4096 | 80 | 80 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  | 1.0000 |
| metadata/cache_reuse/True | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/language/en | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| metadata/language/python | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  | 1.0000 |
| metadata/length_bucket/medium | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| metadata/length_bucket/short | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/metric/classification | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/metric/code | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  | 1.0000 |
| metadata/metric/contains | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/metric/exact_match | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/metric/f1 | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| metadata/needle_position/middle | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/official_task/qasper | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| metadata/official_task/trec | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/question_type/single_hop | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| metadata/source/synthetic | 80 | 80 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  | 1.0000 |
| metadata/task_type/code_qa | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  | 1.0000 |
| metadata/task_type/multi_hop | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/task_type/retrieval | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| metadata/temporal/False | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| model/mock | 80 | 80 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  | 1.0000 |
| overall | 80 | 80 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  | 1.0000 |
| role/primary | 60 | 60 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| role/secondary | 20 | 20 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  | 1.0000 |
| task/LoCoMo/text_qa | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| task/LongBench/classification | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| task/LongBench/single_doc_qa | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  |  |
| task/LongCodeBench/code_qa | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  | 1.0000 |
| task/RULER/multi_hop_trace | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| task/RULER/needle_retrieval | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| task/SCBench/semantic_retrieval | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
| task/SCBench/shared_context_retrieval | 10 | 10 | 0 | 1.0000 | 1.0000 | 1.0000 |  |  |  |
