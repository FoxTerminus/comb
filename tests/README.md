# Tests

Run from the repository root in an environment with the project's dependencies
and pytest installed:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TRITON_CACHE_DIR=/tmp/comb-test-triton \
  python -B -m pytest -q -p no:cacheprovider
```

`pytest.ini` discovers only `tests/` and adds the repository root to the import
path. Importlib mode keeps `tests/training/` and `tests/benchmarks/` from shadowing
the real modules. Both unittest classes and function-style Frozen32 tests are
included (73 tests at the time of the directory reorganization).

- `training/`: data bucketing, checkpoint I/O, resumption, TP helpers and small CPU
  models exercising Frozen32 trainable boundaries, no-context behavior and PIC.
- `benchmarks/`: evaluation helpers, cache audits and mocked server concurrency.

These tests do not establish full-model GPU training or evaluation correctness.
The local `comb-ds0195` environment currently lacks pytest; the 2026-09-09
reorganization checks used its interpreter with the base environment's installed
pytest appended to `sys.path`. No dependency environment was modified.
