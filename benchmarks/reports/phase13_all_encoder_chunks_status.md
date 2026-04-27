# Phase 13: All Encoder Chunks CombLlama Evaluation

## Purpose

This phase changes the benchmark interpretation of CombLlama compression.

Previous compressed sweeps used `compression_ratio < 1.0` to drop history chunks before the chunk encoder. That measured a chunk-retention heuristic, not the intended CombLlama architecture where all chunks go through the encoder and compression comes from the encoder/decoder design.

This phase adds and evaluates:

```text
retention_policy = all_encoder_chunks
```

Under this policy, every historical chunk is sent to the CombLlama chunk encoder. No benchmark-side chunk dropping is applied.

## Code Change

Updated `benchmarks/scripts/prompting.py`:

- `pack_combllama_prompt(...)` now supports `all_encoder_chunks`.
- `inspect_combllama_packing(...)` reports all history chunks as retained under this policy.
- Older chunk-dropping policies have been removed from the runnable benchmark path.

Updated default config:

```json
"retention_policy": "all_encoder_chunks"
```

in `benchmarks/configs/combllama_phase3_sweep.json`.

## Validation

Packing self-check:

```text
compression_ratio 1.0: total chunks 9, kept chunks 9, dropped chunks 0
compression_ratio 0.5: total chunks 9, kept chunks 9, dropped chunks 0
compression_ratio 0.25: total chunks 9, kept chunks 9, dropped chunks 0
```

Static checks:

```bash
/data3/junhaohu/anaconda3/envs/comb/bin/python -m py_compile \
  benchmarks/scripts/prompting.py \
  benchmarks/scripts/run_sweep.py \
  benchmarks/scripts/diagnose_failures.py
```

## Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,LongBench,SCBench,LongCodeBench,LoCoMo \
  --compression-ratios 1.0 \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase13_all_encoder_chunks_dev
```

Scoring and diagnostics:

```bash
PYTHONPATH=/data3/junhaohu/comb /data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/score_predictions.py \
  --manifest benchmarks/results/phase13_all_encoder_chunks_dev/dev_sweep_manifest.json \
  --split dev \
  --output-dir benchmarks/reports/phase13_all_encoder_chunks/scored

PYTHONPATH=/data3/junhaohu/comb /data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/diagnose_failures.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,LongBench,SCBench,LongCodeBench,LoCoMo \
  --compression-ratios 1.0 \
  --predictions-manifest benchmarks/results/phase13_all_encoder_chunks_dev/dev_sweep_manifest.json \
  --output-dir benchmarks/reports/phase13_all_encoder_chunks/failure_analysis
```

## Results

| Benchmark | Examples | Score | Max Memory GB | Prefill s | Decode s | Errors | Collapse | Answer Dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RULER | 12 | 0.167 | 27.01 | 0.496 | 3.439 | 0 | 0.000 | 0.000 |
| LongBench | 5 | 0.140 | 23.11 | 0.191 | 3.569 | 0 | 0.000 | 0.000 |
| SCBench | 4 | 0.119 | 38.46 | 1.932 | 6.470 | 0 | 0.000 | 0.000 |
| LongCodeBench | 3 | 0.000 | 22.48 | 0.051 | 1.426 | 0 | 0.000 | 0.000 |
| LoCoMo | 1 | 0.000 | 24.88 | 0.476 | 1.643 | 0 | 0.000 | 0.000 |

## Interpretation

This is now the correct CombLlama architecture-level baseline for the current benchmark harness: all chunks are encoded, and no evidence is removed by benchmark-side chunk dropping.

The run confirms:

- All 25 current dev examples ran successfully.
- No chunks were dropped.
- No generation collapse was detected.
- Memory and latency match the expected full-context encoder path.

The scores remain modest on this tiny dev subset, so the remaining quality issues are not caused by chunk dropping in this run. They are more likely due to model capability, prompt/task mismatch, local metric limitations, or the small and partially synthetic dev subset.

## Updated Benchmark Semantics

Use `all_encoder_chunks` for fair CombLlama-vs-YOCO comparison.

Do not use chunk-dropping retention policies for the main comparison. They have been removed from the runnable benchmark path.

The previous compressed-ratio sweep should be treated as an ablation of chunk retention, not as the main CombLlama compression result.

## Artifacts

- Predictions: `benchmarks/results/phase13_all_encoder_chunks_dev/dev_predictions_all_encoder_chunks_cr1p0.jsonl`
- Manifest: `benchmarks/results/phase13_all_encoder_chunks_dev/dev_sweep_manifest.json`
- Score summary: `benchmarks/reports/phase13_all_encoder_chunks/scored/benchmark_summary.csv`
- Task summary: `benchmarks/reports/phase13_all_encoder_chunks/scored/task_summary.csv`
- Failure diagnostics: `benchmarks/reports/phase13_all_encoder_chunks/failure_analysis/collapse_summary.csv`
