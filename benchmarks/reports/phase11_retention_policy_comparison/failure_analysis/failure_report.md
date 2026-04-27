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
| keep_early_recent_even | 0.25 | LongBench | 5 | 0.800 | 0.400 | 0.000 |
| keep_early_recent_even | 0.25 | RULER | 12 | 0.167 | 0.750 | 0.250 |
| keep_early_recent_even | 0.25 | SCBench | 4 | 0.000 | 0.000 | 0.500 |
| keep_early_recent_even | 0.5 | LongBench | 5 | 0.200 | 0.200 | 0.200 |
| keep_early_recent_even | 0.5 | RULER | 12 | 0.000 | 0.500 | 0.500 |
| keep_early_recent_even | 0.5 | SCBench | 4 | 0.000 | 0.250 | 0.250 |
| keep_even | 0.25 | LongBench | 5 | 0.000 | 0.200 | 0.200 |
| keep_even | 0.25 | RULER | 12 | 0.000 | 0.667 | 0.333 |
| keep_even | 0.25 | SCBench | 4 | 0.000 | 0.000 | 0.500 |
| keep_even | 0.5 | LongBench | 5 | 0.000 | 0.200 | 0.200 |
| keep_even | 0.5 | RULER | 12 | 0.000 | 0.583 | 0.417 |
| keep_even | 0.5 | SCBench | 4 | 0.000 | 0.250 | 0.250 |

## Notes

A collapse flag is raised for repeated compact substrings, very low token diversity with a dominant token, or long same-token runs.

Packing diagnostics reflect the current benchmark policy: compressed ratios retain the most recent history chunks and drop earlier history chunks.
