# Launch scripts

Shell entrypoints are collected here; Python modules and DeepSpeed configs remain
in `training/` and `benchmarks/`. Invoke scripts with `bash` from the repository
root, for example `bash scripts/training/launch_frozen32_squad.sh` after reviewing
its settings. The scripts select their required working directory.

| Location | Contents |
| --- | --- |
| `training/launch_frozen32_*.sh` | Stage-specific training recipes and milestone evaluation |
| `training/run_frozen32_ni_curated_from_squad.sh` | SQuAD rebuild followed by curated NI |
| `training/watch_frozen32_*.sh` | Historical milestone scheduling and checkpoint watchers |
| `training/launch_llama_repro_true_tp.sh` | Original COMB true TP launcher |
| `training/export_true_tp_checkpoint_to_hf.sh` | Original COMB true TP export |
| `training/run_long_context_capacity_preflight.sh` | Historical long-context diagnostic |
| `benchmarks/run_offline_dataset_repro.sh` | Original offline reproduction launcher |

These are recorded local run recipes, not generic defaults for new experiments.
Several contain fixed GPU IDs, Conda environments, checkpoint paths and step
numbers. In particular, the original NI launch recipe refers to `squad_stage1`,
whose training records have been archived; the curated route uses
`squad_rebuild_for_ni_curated`. The historical NI milestone schedule also assumes
its original evaluation paths. Review and supply appropriate paths before reuse.

Old `training/*.sh` and `benchmarks/*.sh` paths have moved here without compatibility
symlinks. In-repository live references have been updated; external commands or
historical reports may still contain the previous locations.
