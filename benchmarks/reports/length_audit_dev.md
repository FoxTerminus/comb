# Length Audit

Tokenizer: `whitespace`
Records: 80
Prompt tokens: min=149, avg=168.375, max=264

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
| benchmark/LoCoMo | 10 | 158.00 | 158 |
| benchmark/LongBench | 20 | 153.00 | 154 |
| benchmark/LongCodeBench | 10 | 156.00 | 156 |
| benchmark/RULER | 20 | 210.50 | 264 |
| benchmark/SCBench | 20 | 153.00 | 157 |
| bucket/<= 4096 | 80 | 168.38 | 264 |
| task/LoCoMo/text_qa | 10 | 158.00 | 158 |
| task/LongBench/classification | 10 | 154.00 | 154 |
| task/LongBench/single_doc_qa | 10 | 152.00 | 152 |
| task/LongCodeBench/code_qa | 10 | 156.00 | 156 |
| task/RULER/multi_hop_trace | 10 | 157.00 | 157 |
| task/RULER/needle_retrieval | 10 | 264.00 | 264 |
| task/SCBench/semantic_retrieval | 10 | 149.00 | 149 |
| task/SCBench/shared_context_retrieval | 10 | 157.00 | 157 |
