# Phase 15: 50 Examples per Benchmark with All Encoder Chunks

## Scope

This phase runs CombLlama with the corrected benchmark semantics:

```text
retention_policy = all_encoder_chunks
```

Every benchmark contributes 50 examples:

- RULER: 50 examples from a 51-example synthetic dev set.
- LongBench: 50 examples from the expanded dev set.
- SCBench: 50 examples from the expanded dev set.
- LongCodeBench: 50 examples from the expanded dev set.
- LoCoMo: 50 examples from the expanded dev set.

Total final combined rows:

```text
250 examples
250 successful generations
0 errors
0 dropped chunks
0 detected collapse
```

## Data Preparation

The dev sets were rebuilt with:

```bash
PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/prepare_devsets.py \
  --examples-per-source 50 \
  --ruler-lengths 4096,6144,8192,10240,12288,14336,16384,18432,20480,22528,24576,26624,28672,30720,32768,34816,36864 \
  --seed 13 \
  --local-files-only
```

Generated local rows:

| Benchmark | Rows |
| --- | ---: |
| RULER | 51 |
| LongBench | 250 |
| SCBench | 200 |
| LongCodeBench | 150 |
| LoCoMo | 50 |

`benchmarks/scripts/prepare_devsets.py` was fixed so LongCodeBench uses `repo_text` as context instead of only the repository name.

## Runs

Primary run on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,LongBench,SCBench,LongCodeBench,LoCoMo \
  --limit-per-benchmark 50 \
  --compression-ratios 1.0 \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase15_all_encoder_chunks_50_each
```

SCBench initially produced 49 OOM errors on GPU 0 because other processes were already occupying much of the card. SCBench was rerun separately on GPU 5:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks SCBench \
  --limit-per-benchmark 50 \
  --compression-ratios 1.0 \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase15_all_encoder_chunks_scbench50_gpu5
```

The final combined result uses:

- RULER, LongBench, LongCodeBench, LoCoMo from the primary GPU 0 run.
- SCBench from the successful GPU 5 rerun.

Final combined manifest:

```text
benchmarks/results/phase15_all_encoder_chunks_50_each_combined/dev_sweep_manifest.json
```

## Final Benchmark Summary

| Benchmark | Examples | Score | Max Memory GB | Prefill s | Decode s | Errors | Collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RULER | 50 | 0.040 | 27.60 | 0.547 | 3.929 | 0 | 0.000 |
| LongBench | 50 | 0.137 | 23.05 | 0.177 | 3.139 | 0 | 0.000 |
| SCBench | 50 | 0.000 | 38.51 | 3.048 | 9.349 | 0 | 0.000 |
| LongCodeBench | 50 | 0.700 | 25.76 | 0.486 | 3.992 | 0 | 0.000 |
| LoCoMo | 50 | 0.160 | 24.90 | 0.478 | 3.844 | 0 | 0.000 |

## Task Summary

| Benchmark | Task | Examples | Metric | Score |
| --- | --- | ---: | --- | ---: |
| RULER | passkey_retrieval_early | 17 | passkey_contains | 0.059 |
| RULER | passkey_retrieval_middle | 17 | passkey_contains | 0.000 |
| RULER | passkey_retrieval_late | 16 | passkey_contains | 0.062 |
| LongBench | qasper_e | 50 | qa_f1 | 0.137 |
| SCBench | scbench_kv | 50 | exact_or_contains | 0.000 |
| LongCodeBench | LongCodeQA_32K | 50 | choice_match | 0.700 |
| LoCoMo | multi_hop | 24 | choice_match | 0.083 |
| LoCoMo | single_hop | 19 | choice_match | 0.316 |
| LoCoMo | temporal_reasoning | 7 | choice_match | 0.000 |

## Interpretation

The infrastructure now successfully runs 50 examples per benchmark under the corrected all-encoder-chunks policy.

Quality is uneven:

- LongCodeBench is strongest at `0.700` choice accuracy on the first 50 LongCodeQA 32K examples.
- LoCoMo is modest at `0.160`, with single-hop better than multi-hop and temporal reasoning.
- LongBench Qasper QA remains low at `0.137` QA F1.
- RULER passkey retrieval is weak at `0.040`, especially middle-position retrieval.
- SCBench KV retrieval scores `0.000`; the model produces UUID-like values but not exact requested values.

The important stability result is positive:

- No benchmark-side evidence dropping.
- No runtime errors in the final combined result.
- No detected repetitive generation collapse.

## Caveats

These are still local diagnostic metrics, not official benchmark leaderboard scores.

The selected 50 examples per benchmark are deterministic but not fully balanced:

- LongBench first 50 are all `qasper_e`.
- LongCodeBench first 50 are all `LongCodeQA_32K`.
- SCBench first 50 are all `scbench_kv`.
- LoCoMo includes multi-hop, single-hop, and temporal reasoning.
- RULER is a generated synthetic sweep over length and position.

SCBench full-context all-encoder evaluation requires a mostly free 80GB GPU. GPU 0 failed due to memory pressure from other resident processes; GPU 5 completed the same 50 examples successfully.

## Artifacts

- Primary predictions: `benchmarks/results/phase15_all_encoder_chunks_50_each/`
- SCBench GPU5 rerun: `benchmarks/results/phase15_all_encoder_chunks_scbench50_gpu5/`
- Final combined predictions: `benchmarks/results/phase15_all_encoder_chunks_50_each_combined/dev_predictions_all_encoder_chunks_cr1p0.jsonl`
- Final scored report: `benchmarks/reports/phase15_all_encoder_chunks_50_each_combined/scored/score_report.md`
- Final benchmark summary: `benchmarks/reports/phase15_all_encoder_chunks_50_each_combined/scored/benchmark_summary.csv`
- Final task summary: `benchmarks/reports/phase15_all_encoder_chunks_50_each_combined/scored/task_summary.csv`
- Final failure diagnostics: `benchmarks/reports/phase15_all_encoder_chunks_50_each_combined/failure_analysis/failure_report.md`
