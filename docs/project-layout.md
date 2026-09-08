# Comb project guide

## Source tree

| Directory | Responsibility |
| --- | --- |
| `comb/` | Model implementations, serving entrypoints, PIC storage and transfer |
| `data/` | Dataset adapters, curation, tokenization and bucket caching |
| `training/` | Training, checkpoint/export/validation helpers and DeepSpeed configs |
| `benchmarks/` | Offline/online benchmarks and local evaluation/audit tools |
| `scripts/` | Shell launchers, grouped by training and benchmarks |
| `tests/` | CPU regression tests, grouped by the modules they exercise |
| `examples/` | Original COMB usage examples |
| `assets/` | README assets |
| `docs/archive/` | Historical reports; not current run instructions |

Python module locations remain stable because trainers, evaluators and tests share
them. Training config filenames and locations remain stable for CLI compatibility
and source snapshot collection. Test and shell files are separated from those
modules, without duplicate forwarding files.

## Model paths

The original COMB HF implementation is `comb/integration/hf/CombLlama.py`, with
serving integration under `comb/integration/vllm/`. The local Frozen32 model is
`comb/integration/hf/CombLlamaFrozen32.py`, trained through
`training/train_llama_frozen32.py` and the shared reproduction loop. These model
families and their exports are not interchangeable.

Frozen32 freezes the 32-layer encoder and decoder backbone and trains the context
branches at decoder layers 3, 7, 11, 15, 19, 23, 27 and 31. Its context MLP is on
the context branch, with scalar tanh gates. Native checkpoints contain TP shards;
HF exports reconstruct the model for HF evaluation. An HF export is not a native
optimizer-resume checkpoint.

## Current local checkpoint layout

Weights live outside Git at `/data3/junhaohu/checkpoints/CombLlamaFrozen32/`.
The following directory names identify distinct training histories; they should
not be treated as duplicate weights merely because the architecture is the same.

| Directory | Training history / retained milestone |
| --- | --- |
| `squad_rebuild_for_ni_curated/` | Rebuilt SQuAD stage, step 4,073 |
| `ni_curated_from_squad/` | Rebuilt SQuAD → curated NI, step 22,279 |
| `ni_stage2/` | Original SQuAD → full NI, step 196,690 |
| `xsum_stage3/` | Full NI → XSum, step 203,068 |
| `superni_curated_from_ni/` | Full NI → curated SuperNI, step 205,965 |
| `superni_full_from_ni/` | Full NI → full SuperNI, step 258,907 |
| `superni_stage4/` | XSum → full SuperNI, step 265,285 |

The current curated-NI HF export is
`bench_22279_ni_curated_longbench_full/export_work/hf_step_00022279/` under that
checkpoint root. Historical evaluation outputs, run logs and training records
are under its `_archive/`. Older non-Frozen32 checkpoints are under
`/data3/junhaohu/checkpoints/_archive/legacy_comb/`.

These are local paths, not files to add to Git. Retain run configs, dataset
manifests and provenance alongside weights. A curated NI run and the full NI run
differ in both starting weights and training budget, so their score difference
alone does not isolate the effect of data filtering.

## Historical tools

Unique historical Python tools are retained rather than deleted on the basis of
age. Several are useful for inspecting native shards, validating HF exports and
comparing frozen/trainable tensors. Known incomplete historical dependencies:

- `supervise_single_training.py` expects the missing `run_full_resume_single.sh`.
- `supervise_true_tp_training.py` expects the missing `run_full_resume_true_tp.sh`.
- `preflight_true_tp_long_context.py` refers to the missing `verify_tp2_checkpoint.py`.

These missing files predate this reorganization. The tools are not advertised as
ready-to-run current launchers. The archived reproduction report preserves the
original paths and status assertions as historical evidence.

## Generated files and Git

Keep weights, logs and experiment outputs outside the source tree. `outputs/`
is available as an ignored local scratch directory; existing `benchmarks/results/`
remains ignored. Shared reproduction training defaults TensorBoard logs to the
run's output directory and respects an explicit config override. TensorBoard event
files and Python caches are ignored, while source, tests and configs remain visible.

The September 9 worktree reorganization backed up all existing source files and
the initial tracked diff under
`/data3/junhaohu/checkpoints/_archive/organization_records/comb_worktree_20260909_003431/`.
That directory also records source moves and holds the 90 archived TensorBoard
event files. No model weights were moved by the source-tree reorganization.

See [tests/README.md](../tests/README.md) for CPU validation and
[the Git workflow](git-workflow.md) for maintaining the development baseline.
Source, configuration and tests belong in version control; generated artifacts
do not. Historical tools with known missing dependencies remain explicitly labeled.
