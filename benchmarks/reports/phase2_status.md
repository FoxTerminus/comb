# Phase 2 Dev Subset Status

## Scope

Phase 2 focuses on CombLlama only. YOCO is intentionally excluded because it is still training.

The dev subset covers:

1. RULER synthetic passkey retrieval at 4K, 8K, 16K, and 32K word-level contexts.
2. LongBench-E cached subsets: `qasper_e`, `hotpotqa_e`, `gov_report_e`, `lcc_e`, `repobench-p_e`.
3. SCBench cached subsets: `scbench_kv`, `scbench_repoqa`, `scbench_qa_eng`, `scbench_summary`.
4. LongCodeBench fallback parser from cached `Steefano/LCB` `LongCodeQA.zip`: 32K, 64K, 128K.
5. LoCoMo-MC10 fallback parser from cached `Percena/locomo-mc10` JSONL.

## Generated Data

```text
benchmarks/RULER/dev.jsonl              12 examples
benchmarks/LongBench/dev.jsonl           5 examples
benchmarks/SCBench/dev.jsonl             4 examples
benchmarks/LongCodeBench/dev.jsonl       3 examples
benchmarks/LoCoMo/dev.jsonl              1 example
```

Manifest:

```text
benchmarks/results/phase2_dev_manifest.json
```

## Real CombLlama Runs

Primary run:

```text
benchmarks/results/phase2_dev_combllama/dev_predictions.jsonl
benchmarks/reports/phase2_dev_combllama_summary.csv
```

Repeat run:

```text
benchmarks/results/phase2_dev_combllama_repeat/dev_predictions.jsonl
benchmarks/reports/phase2_dev_combllama_repeat_summary.csv
```

Both runs used:

```text
CUDA_VISIBLE_DEVICES=0
max_new_tokens=8
config=benchmarks/configs/combllama_dev.json
```

## Validation

```text
records: 25
errors: 0
missing_memory: 0
missing_latency: 0
prediction mismatches between primary and repeat: 0
benchmarks covered: LoCoMo, LongBench, LongCodeBench, RULER, SCBench
```

Runtime range from the primary run:

```text
chunk_tokens_min_max: 0 to 123048
decoder_tokens_min_max: 163 to 1024
peak_memory_min_max: 22.47 GB to 38.47 GB
prefill_latency_min_max: 0.03 s to 2.46 s
decode_latency_min_max: 0.20 s to 17.23 s
```

## Caveats

1. `max_new_tokens=8` is for infrastructure validation only. The exact-match scores are not meaningful yet because many answers are longer than 8 tokens.
2. LongCodeBench and LoCoMo required direct cached-file parsers because their HuggingFace dataset builders failed on mixed schemas.
3. SCBench uses the first turn in `multi_turns` for each selected shared-context example.
4. LongCodeBench currently uses LongCodeQA only; LongSWE-Bench repair evaluation is not enabled in this phase.
5. Some examples have no compressed chunk path because the prompt fits entirely inside the recent decoder window; full evaluation should use longer examples or reduce `recent_window_tokens` when the goal is to stress chunk compression.
