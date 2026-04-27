# Failure Diagnostics

## Outputs

```text
rendered_prompts.jsonl
packing_diagnostics.jsonl
prediction_diagnostics.jsonl
collapse_summary.csv
```

## Collapse Summary

| Policy | Compression | Benchmark | Examples | Collapse Rate | Answer Dropped | Answer Kept/Decoder |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| all_encoder_chunks | 1.0 | LoCoMo | 1 | 0.000 | 0.000 | 1.000 |
| all_encoder_chunks | 1.0 | LongBench | 5 | 0.000 | 0.000 | 0.400 |
| all_encoder_chunks | 1.0 | LongCodeBench | 3 | 0.000 | 0.000 | 1.000 |
| all_encoder_chunks | 1.0 | RULER | 12 | 0.000 | 0.000 | 1.000 |
| all_encoder_chunks | 1.0 | SCBench | 4 | 0.000 | 0.000 | 0.500 |

## Notes

A collapse flag is raised for repeated compact substrings, very low token diversity with a dominant token, or long same-token runs.

Packing diagnostics reflect the current benchmark policy: compressed ratios retain the most recent history chunks and drop earlier history chunks.
