# Length Audit

Tokenizer: `whitespace`
Records: 8
Prompt tokens: min=49, avg=55.875, max=64

| Threshold | Count Over |
| ---: | ---: |
| 4096 | 0 |
| 8192 | 0 |
| 16384 | 0 |
| 32768 | 0 |
| 65536 | 0 |
| 131072 | 0 |

| Model | Count Over Limit |
| --- | ---: |
| llama | 0 |
| combllama | 0 |
| yoco | 0 |
| sambay | 0 |

| Group | Count | Avg Prompt Tokens | Max Prompt Tokens |
| --- | ---: | ---: | ---: |
| benchmark/LoCoMo | 1 | 58.00 | 58 |
| benchmark/LongBench | 2 | 53.00 | 54 |
| benchmark/LongCodeBench | 1 | 56.00 | 56 |
| benchmark/RULER | 2 | 60.50 | 64 |
| benchmark/SCBench | 2 | 53.00 | 57 |
| bucket/<= 4096 | 8 | 55.88 | 64 |
| task/LoCoMo/text_qa | 1 | 58.00 | 58 |
| task/LongBench/classification | 1 | 54.00 | 54 |
| task/LongBench/single_doc_qa | 1 | 52.00 | 52 |
| task/LongCodeBench/code_qa | 1 | 56.00 | 56 |
| task/RULER/multi_hop_trace | 1 | 57.00 | 57 |
| task/RULER/needle_retrieval | 1 | 64.00 | 64 |
| task/SCBench/semantic_retrieval | 1 | 49.00 | 49 |
| task/SCBench/shared_context_retrieval | 1 | 57.00 | 57 |
