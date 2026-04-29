# CombLlama Benchmarks

This is the rebuilt benchmark framework for comparing CombLlama with Llama, YOCO, and SambaY on long-context tasks.

Quick smoke test:

```bash
python -m benchmarks.scripts.runners.run_smoke --model mock
```

Generic runner:

```bash
python -m benchmarks.scripts.runners.run_eval \
  --run-config benchmarks/configs/runs/smoke_mock.json
```

Prepare synthetic smoke/dev data:

```bash
python -m benchmarks.scripts.converters.prepare_data --splits smoke dev
```

Convert local official data after placing raw files under `benchmarks/data/raw`:

```bash
python -m benchmarks.scripts.converters.convert_raw \
  --benchmarks RULER SCBench LongBench \
  --split dev \
  --allow-missing
```

Validate converted data without running a model:

```bash
python -m benchmarks.scripts.converters.validate_data --split dev
```

Audit prompt lengths without loading any model:

```bash
python -m benchmarks.scripts.auditing.length_audit --split dev
```

By default the audit falls back to whitespace counting if `transformers` or the tokenizer is unavailable. To require the real tokenizer, run it from an environment with `transformers` installed and add:

```bash
python -m benchmarks.scripts.auditing.length_audit --split dev --require-tokenizer
```

Check real-model adapter readiness without loading weights:

```bash
python -m benchmarks.scripts.adapters.diagnostics
```

Check a specific run config:

```bash
python -m benchmarks.scripts.adapters.diagnostics \
  --run-config benchmarks/configs/runs/smoke_combllama.json
```

The diagnostics command is expected to fail clearly when a checkpoint is missing, `transformers`/`torch` is not installed, or a config requests CUDA in a CPU-only environment. It does not require YOCO or SambaY to be trained.

Summarize failures from a completed run:

```bash
python -m benchmarks.scripts.reporting.failures \
  --run-dir benchmarks/results/dev/dev_mock_latest
```

Build a retry manifest for failed samples:

```bash
python -m benchmarks.scripts.runners.retry \
  --run-dir benchmarks/results/dev/dev_mock_latest
```

Retry only failed samples from a previous run:

```bash
python -m benchmarks.scripts.runners.run_eval \
  --run-config benchmarks/configs/runs/dev_mock.json \
  --run-id dev_mock_retry \
  --retry-from benchmarks/results/dev/dev_mock_latest
```

Validate an environment snapshot:

```bash
python -m benchmarks.scripts.utils.environment \
  --environment benchmarks/results/dev/dev_mock_latest/environment.json
```
