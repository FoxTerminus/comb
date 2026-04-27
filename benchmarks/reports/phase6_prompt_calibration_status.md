# Phase 6 Prompt Calibration Status

## Scope

This phase calibrates benchmark-specific prompts for CombLlama and reruns the
same tiny dev sweep used in Phase 4/5. The goal is to reduce avoidable scoring
loss from generic prompts before increasing benchmark sample counts.

## Prompt Changes

The shared prompt renderer is now benchmark-aware:

```text
RULER: ask for passkey-only output
LongBench QA/retrieval: ask for short exact answers
LongBench summarization: use summary-specific request wording
LongBench code tasks: ask for code-only/completion-oriented output
SCBench KV: ask for exact value retrieval
SCBench repo/code QA: ask for code or identifier only
SCBench summary: use summary-specific wording
LongCodeBench choices: ask for option letter
LoCoMo choices: ask for exact answer text from options
```

The choice parser was also fixed to support `A-Z` rather than only `A-D`, which
is required for LoCoMo-MC10.

## Validation

Static checks passed:

```bash
/data3/junhaohu/anaconda3/envs/comb/bin/python -m py_compile \
  benchmarks/scripts/prompting.py \
  benchmarks/scripts/score_predictions.py
```

Full prompt-calibrated dev sweep:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data3/junhaohu/comb \
/data3/junhaohu/anaconda3/envs/comb/bin/python benchmarks/scripts/run_sweep.py \
  --config benchmarks/configs/combllama_phase3_sweep.json \
  --split dev \
  --benchmarks RULER,LongBench,SCBench,LongCodeBench,LoCoMo \
  --compression-ratios 1.0,0.5,0.25,0.125 \
  --max-new-tokens 32 \
  --output-dir benchmarks/results/phase6_combllama_dev_sweep_prompt_calibrated
```

LoCoMo prompt checks were also run separately after fixing the 10-choice prompt:

```text
benchmarks/results/phase6_locomo_choice_prompt_check
benchmarks/results/phase6_locomo_answer_text_prompt_check
```

Both LoCoMo variants scored 0 on the current single dev example, indicating that
the failure is not only a letter-vs-text formatting problem.

## Outputs

```text
benchmarks/results/phase6_combllama_dev_sweep_prompt_calibrated/
benchmarks/reports/phase6_prompt_calibration/scored/
benchmarks/reports/phase6_prompt_calibration/analysis/
benchmarks/reports/phase6_prompt_calibration/sweep_summary.csv
benchmarks/reports/phase6_prompt_calibration/phase5_vs_phase6_benchmark_delta.csv
```

## Phase 5 vs Phase 6

For `compression_ratio=1.0`:

```text
RULER:       0.083 -> 0.167  (+0.083)
SCBench:     0.104 -> 0.119  (+0.015)
LongBench:   0.154 -> 0.140  (-0.014)
LongCode:    0.000 -> 0.000  (+0.000)
LoCoMo:      1.000 -> 0.000  (-1.000)
```

For compressed ratios, prompt calibration did not change the main conclusion:
generation still collapses into repetitive tokens such as `self`, `Piet`, commas,
or periods on many examples.

## Interpretation

Prompt calibration helped RULER and slightly improved SCBench at full context,
but it did not fix compressed-generation collapse. It also exposed that the
single LoCoMo dev example is unstable under prompt changes; this tiny one-example
LoCoMo number should not be used as a final claim.

The current strongest technical conclusion remains:

```text
CombLlama full context can recover some long-context facts on the tiny dev set,
but the current simulated compression strategy sharply degrades generation
quality for natural-language tasks.
```

## Recommended Next Step

Before scaling to full benchmark runs, do a small failure-analysis phase:

```text
1. Inspect prompt/rendered-token packing for compressed ratios.
2. Verify whether keeping most-recent chunks is the right compression proxy.
3. Add a repetition/collapse diagnostic metric.
4. Add per-example rendered prompt dumps for official reproducibility.
5. Decide whether to compare only full-context CombLlama first, or to fix the
   compression path before spending GPU time on larger compressed sweeps.
```
