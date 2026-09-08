# Git development baseline

## What belongs in Git

Track model implementations, dataset adapters and curation code, training and
evaluation modules, launch recipes, configs, tests, and documentation. Launch
recipes record local environments and paths; check those settings before reuse.

Keep weights, dataset caches, logs, predictions, and generated reports outside
the repository. Existing `benchmarks/results/` and local `outputs/` are ignored.
Do not hide new Python files or entire source directories in `.gitignore`.

## Baseline commits

The local development baseline separates three changes:

1. Existing COMB/Frozen32 implementation, dataset curation, training/export tools,
   evaluation helpers, regression tests and historical reproduction notes. This
   preserves the source snapshot from before directory reorganization.
2. Test and shell directories, documentation, generated-file ignore rules, and
   references updated for moved entrypoints. Python module and config paths stay
   stable. The curriculum supervisor also resolves launchers under `scripts/`.
3. Enabled TensorBoard logging in the shared training loop defaults to the run
   output directory; an explicit config path takes precedence.

Recording existing work does not certify every historical launcher as usable or
every evaluation protocol as a sealed benchmark. Known missing legacy dependencies
are listed in [the project guide](project-layout.md#historical-tools). CPU test
coverage is described in [the test guide](../tests/README.md).

## Ongoing work

Keep each new change focused on one behavior or structural concern. Before a
commit, inspect both staged and unstaged content:

```bash
git status --short
git diff
git diff --cached
git diff --check
```

Stage explicit paths so unrelated experiment outputs are not accidentally added.
Run the relevant checks, then create a local commit with its purpose and validation
in the message. A clean `git status` means tracked work matches the baseline; it
does not imply that ignored run artifacts were deleted.

Use a separate branch or worktree for experiments that change source concurrently
with a stable training run. Record the commit and any remaining source diff with
each run, alongside its config and dataset manifest. Do not rewrite an old run's
provenance after moving scripts or updating launch defaults.
