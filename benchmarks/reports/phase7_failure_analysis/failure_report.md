# Phase 7 Failure Diagnostics

## Outputs

```text
rendered_prompts.jsonl
packing_diagnostics.jsonl
prediction_diagnostics.jsonl
collapse_summary.csv
```

## Collapse Summary

| Compression | Benchmark | Examples | Collapse Rate | Answer Dropped | Answer Kept/Decoder |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.125 | LoCoMo | 1 | 1.000 | 0.000 | 1.000 |
| 0.125 | LongBench | 5 | 0.800 | 0.400 | 0.000 |
| 0.125 | LongCodeBench | 3 | 0.000 | 0.000 | 1.000 |
| 0.125 | RULER | 12 | 0.917 | 0.667 | 0.333 |
| 0.125 | SCBench | 4 | 1.000 | 0.500 | 0.000 |
| 0.25 | LoCoMo | 1 | 1.000 | 0.000 | 1.000 |
| 0.25 | LongBench | 5 | 0.800 | 0.400 | 0.000 |
| 0.25 | LongCodeBench | 3 | 0.000 | 0.000 | 1.000 |
| 0.25 | RULER | 12 | 0.917 | 0.667 | 0.333 |
| 0.25 | SCBench | 4 | 0.750 | 0.500 | 0.000 |
| 0.5 | LoCoMo | 1 | 1.000 | 0.000 | 1.000 |
| 0.5 | LongBench | 5 | 0.800 | 0.400 | 0.000 |
| 0.5 | LongCodeBench | 3 | 0.000 | 0.000 | 1.000 |
| 0.5 | RULER | 12 | 1.000 | 0.500 | 0.500 |
| 0.5 | SCBench | 4 | 0.750 | 0.250 | 0.250 |
| 1.0 | LoCoMo | 1 | 0.000 | 0.000 | 1.000 |
| 1.0 | LongBench | 5 | 0.000 | 0.000 | 0.400 |
| 1.0 | LongCodeBench | 3 | 0.000 | 0.000 | 1.000 |
| 1.0 | RULER | 12 | 0.000 | 0.000 | 1.000 |
| 1.0 | SCBench | 4 | 0.000 | 0.000 | 0.500 |

## Notes

A collapse flag is raised for repeated compact substrings, very low token diversity with a dominant token, or long same-token runs.

Packing diagnostics reflect the current benchmark policy: compressed ratios retain the most recent history chunks and drop earlier history chunks.
