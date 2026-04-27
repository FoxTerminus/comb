# Phase 3 Status

## Completed

Phase 3 infrastructure is implemented for CombLlama-only evaluation:

1. Compression-ratio sweep runner.
2. Per-ratio raw prediction outputs.
3. Sweep manifest.
4. Aggregate sweep summary with quality and efficiency metrics.
5. CLI overrides for compression ratio, chunk size, recent decoder window, and generation length.

New files:

```text
benchmarks/configs/combllama_phase3_sweep.json
benchmarks/scripts/metrics.py
benchmarks/scripts/run_sweep.py
benchmarks/scripts/summarize_sweep.py
```

Updated files:

```text
benchmarks/scripts/run_examples.py
benchmarks/scripts/summarize_results.py
```

## Real Sweep Smoke

Command scope:

```text
benchmark: RULER
split: dev
examples: 3
compression_ratios: 1.0, 0.5, 0.25
max_new_tokens: 16
gpu: CUDA_VISIBLE_DEVICES=0
```

Outputs:

```text
benchmarks/results/phase3_combllama_ruler_sweep_smoke/dev_predictions_cr1p0.jsonl
benchmarks/results/phase3_combllama_ruler_sweep_smoke/dev_predictions_cr0p5.jsonl
benchmarks/results/phase3_combllama_ruler_sweep_smoke/dev_predictions_cr0p25.jsonl
benchmarks/results/phase3_combllama_ruler_sweep_smoke/dev_sweep_manifest.json
benchmarks/reports/phase3_combllama_ruler_sweep_smoke_summary.csv
```

Validation:

```text
ratio 1.0: 3 rows, 0 errors, chunk_tokens=3578, peak_memory=22.91 GB
ratio 0.5: 3 rows, 0 errors, chunk_tokens=1530, peak_memory=22.64 GB
ratio 0.25: 3 rows, 0 errors, chunk_tokens=506, peak_memory=22.60 GB
```

This confirms that the sweep captures both efficiency changes and quality changes as the retained chunk context is reduced.

## Not Yet Completed

The full Phase 3 dev matrix has now been run with `max_new_tokens=8`. A first
attempt with `max_new_tokens=32` was stopped because the current conservative
adapter re-runs the full packed context for every generated token and was too
slow for full-matrix execution.

Full dev sweep outputs:

```text
benchmarks/results/phase3_combllama_dev_sweep_mnt8/dev_predictions_cr1p0.jsonl
benchmarks/results/phase3_combllama_dev_sweep_mnt8/dev_predictions_cr0p5.jsonl
benchmarks/results/phase3_combllama_dev_sweep_mnt8/dev_predictions_cr0p25.jsonl
benchmarks/results/phase3_combllama_dev_sweep_mnt8/dev_predictions_cr0p125.jsonl
benchmarks/results/phase3_combllama_dev_sweep_mnt8/dev_sweep_manifest.json
benchmarks/reports/phase3_combllama_dev_sweep_mnt8_summary.csv
```

Full dev sweep validation:

```text
compression ratios: 1.0, 0.5, 0.25, 0.125
examples per ratio: 25
total prediction rows: 100
errors: 0
summary rows: 64 plus header
```

Aggregate efficiency trend:

```text
ratio 1.0:   avg chunk tokens 23933.1, max memory 38.466 GB, avg prefill 0.570 s, avg decode 3.902 s
ratio 0.5:   avg chunk tokens 11563.2, max memory 30.332 GB, avg prefill 0.320 s, avg decode 2.247 s
ratio 0.25:  avg chunk tokens  5542.0, max memory 26.333 GB, avg prefill 0.205 s, avg decode 1.439 s
ratio 0.125: avg chunk tokens  2715.8, max memory 24.334 GB, avg prefill 0.150 s, avg decode 1.057 s
```

The next full-quality command should only be run after the generation path is optimized with KV reuse, or restricted to a small benchmark/task subset:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,LongBench,SCBench,LongCodeBench,LoCoMo \
  --compression-ratios 1.0,0.5,0.25,0.125 \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase3_combllama_dev_sweep
```

Then summarize:

```bash
PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/summarize_sweep.py \
  benchmarks/results/phase3_combllama_dev_sweep/dev_sweep_manifest.json \
  --output-csv benchmarks/reports/phase3_combllama_dev_sweep_summary.csv
```

## Caveats

1. The current compression sweep simulates reduced compressed context by retaining fewer historical chunks before the decoder. It does not yet modify a low-level KV compressor inside CombLlama.
2. The full dev sweep uses `max_new_tokens=8`, so it validates the complete matrix and efficiency curves but is not a final-quality benchmark.
3. `contains_match` is more informative than strict exact match for short generated answers, but official benchmark metrics still need to be plugged in for final reporting.
4. The current adapter is deliberately conservative and re-runs context on each generated token; Phase 4 should optimize generation with cache reuse before long-answer full evaluation.
