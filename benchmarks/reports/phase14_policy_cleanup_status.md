# Phase 14: Remove Chunk-Dropping Policies

## Scope

The runnable benchmark path now supports only:

```text
retention_policy = all_encoder_chunks
```

The old chunk-dropping policies have been removed from active code paths:

```text
keep_recent
keep_early
keep_even
keep_early_recent_even
oracle_keep_answer
```

## Code Changes

Updated `benchmarks/scripts/prompting.py`:

- Removed chunk-selection implementations for the old policies.
- `pack_combllama_prompt(...)` always sends every history chunk to the encoder.
- `inspect_combllama_packing(...)` always reports zero dropped chunks under the supported policy.
- Any unsupported `retention_policy` now raises `ValueError`.

Updated `benchmarks/scripts/run_sweep.py`:

- Removed `--retention-policies`.
- Sweep runs only the config-defined `all_encoder_chunks` policy.

Updated `benchmarks/scripts/diagnose_failures.py`:

- Removed `--retention-policies`.
- Diagnostics are generated only for `all_encoder_chunks`.

Updated `benchmarks/scripts/adapters.py`:

- CombLlama adapter rejects any policy other than `all_encoder_chunks`.

Updated docs:

- `benchmarks/PLAN.md`
- `benchmarks/reports/phase13_all_encoder_chunks_status.md`

## Validation

Static checks passed:

```bash
/data3/junhaohu/anaconda3/envs/comb/bin/python -m py_compile \
  benchmarks/scripts/prompting.py \
  benchmarks/scripts/adapters.py \
  benchmarks/scripts/run_sweep.py \
  benchmarks/scripts/diagnose_failures.py \
  benchmarks/scripts/score_predictions.py
```

Mock sweep with the supported policy passed:

```text
Running retention_policy=all_encoder_chunks compression_ratio=1.0 on 1 examples
OK RULER/retrieval/ruler_smoke_retrieval: 'redwood'
```

Old policy CLI usage is rejected:

```text
run_sweep.py: error: unrecognized arguments: --retention-policies keep_even
```

## Current Semantics

All CombLlama benchmark runs now evaluate the intended architecture-level path:

```text
all history chunks -> chunk encoder -> decoder cross-attention
```

No benchmark-side chunk dropping remains in the active evaluation code.
