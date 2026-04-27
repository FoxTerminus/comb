# Phase 7 Failure Analysis Status

## Scope

This phase does not run model inference. It diagnoses existing Phase 6
CombLlama predictions and the corresponding rendered prompts/packing metadata.

Input manifest:

```text
benchmarks/results/phase6_combllama_dev_sweep_prompt_calibrated/dev_sweep_manifest.json
```

## What Changed

`benchmarks/scripts/prompting.py` now exposes packing diagnostics for the current
CombLlama benchmark policy.

New script:

```text
benchmarks/scripts/diagnose_failures.py
```

The script writes:

```text
benchmarks/reports/phase7_failure_analysis/rendered_prompts.jsonl
benchmarks/reports/phase7_failure_analysis/packing_diagnostics.jsonl
benchmarks/reports/phase7_failure_analysis/prediction_diagnostics.jsonl
benchmarks/reports/phase7_failure_analysis/collapse_summary.csv
benchmarks/reports/phase7_failure_analysis/failure_report.md
```

## Diagnostics Added

For each rendered prompt:

```text
prompt hash
full rendered prompt
encoded prompt length
```

For each compression ratio:

```text
history tokens
decoder tokens
total chunks
kept chunks
dropped chunks
kept/dropped chunk tokens
gold answer token region: decoder, kept_chunk, dropped_chunk, not_found, no_answer
```

For each prediction:

```text
unique token ratio
dominant token fraction
max repeated-token run
repeated compact substring unit
collapse flag
```

## Key Results

Full context does not show collapse in this dev set:

```text
ratio 1.0 collapse rate:
  RULER 0.000
  LongBench 0.000
  SCBench 0.000
  LongCodeBench 0.000
  LoCoMo 0.000
```

Compressed ratios show systematic collapse:

```text
ratio 0.5 collapse rate:
  RULER 1.000
  LongBench 0.800
  SCBench 0.750
  LoCoMo 1.000
  LongCodeBench 0.000

ratio 0.25 collapse rate:
  RULER 0.917
  LongBench 0.800
  SCBench 0.750
  LoCoMo 1.000
  LongCodeBench 0.000

ratio 0.125 collapse rate:
  RULER 0.917
  LongBench 0.800
  SCBench 1.000
  LoCoMo 1.000
  LongCodeBench 0.000
```

The current compression policy also drops some answers:

```text
ratio 0.5 answer dropped rate:
  RULER 0.500
  LongBench 0.400
  SCBench 0.250

ratio 0.25 / 0.125 answer dropped rate:
  RULER 0.667
  LongBench 0.400
  SCBench 0.500
```

However, answer dropping is not the whole issue. Some examples collapse even
when the answer remains in `decoder` or `kept_chunk`. For example, RULER late
passkeys remain in the decoder across compressed ratios, but compressed outputs
still frequently collapse into repeated strings such as `selfself...`.

## Interpretation

There are two separate failure modes:

```text
1. Retrieval loss:
   The current simulated compression policy keeps the most recent history
   chunks and can drop early/middle evidence.

2. Generation instability:
   Compressed contexts trigger repetitive output even when the answer is still
   available in retained context or decoder tokens.
```

This means larger compressed benchmark sweeps would mostly measure the current
compression simulation failure, not a reliable final CombLlama-vs-YOCO result.

## Recommended Next Step

Before scaling sample counts, run a controlled compression-ablation phase:

```text
1. Add retention policies: keep_recent, keep_early, keep_even, and optionally oracle_keep_answer.
2. Run only RULER + SCBench small subsets under those policies.
3. Compare collapse rate and answer-region coverage.
4. If collapse persists even with oracle_keep_answer, prioritize fixing the
   compressed generation path rather than expanding benchmark size.
```

For final reporting right now, the safest claim is:

```text
CombLlama full-context evaluation infrastructure works on the dev subset.
The current simulated compressed-context path is not yet reliable enough for
final benchmark claims.
```
