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
| keep_early | 0.5 | RULER | 3 | 0.000 | 0.333 | 0.667 |
| keep_early | 0.5 | SCBench | 3 | 0.000 | 0.333 | 0.333 |
| keep_even | 0.5 | RULER | 3 | 0.000 | 0.333 | 0.667 |
| keep_even | 0.5 | SCBench | 3 | 0.000 | 0.333 | 0.333 |
| keep_recent | 0.5 | RULER | 3 | 1.000 | 0.333 | 0.667 |
| keep_recent | 0.5 | SCBench | 3 | 0.667 | 0.333 | 0.333 |
| oracle_keep_answer | 0.5 | RULER | 3 | 0.667 | 0.000 | 1.000 |
| oracle_keep_answer | 0.5 | SCBench | 3 | 0.667 | 0.000 | 0.667 |

## Notes

A collapse flag is raised for repeated compact substrings, very low token diversity with a dominant token, or long same-token runs.

Packing diagnostics reflect the current benchmark policy: compressed ratios retain the most recent history chunks and drop earlier history chunks.
