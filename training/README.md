# Train a Comb model

## Current local entrypoints

| Purpose | Entry point |
| --- | --- |
| Frozen32 training | `train_llama_frozen32.py` |
| Shared resume, checkpoint retention, evaluation and export loop | `train_llama_repro.py` |
| Original COMB true tensor parallel training | `train_llama_true_tp_repro.py` |
| Original upstream training | `train_llama.py` |
| Shell launch recipes | [`../scripts/training/`](../scripts/training/) |
| Unit tests | [`../tests/training/`](../tests/training/) |

Keep direct Python training invocations in this directory, with the repository
root on `PYTHONPATH`. The training loop resolves config and source snapshot paths
relative to this working directory. Shell launchers select the working directory
themselves. Their output, resume, GPU and environment paths describe specific
local experiments; inspect them before starting a new run.

DeepSpeed configs remain beside the Python entrypoints to preserve their existing
CLI and source snapshot paths:

| Config | Role |
| --- | --- |
| `ds_llama_frozen32_tp4_stage0.json` | Frozen32 SQuAD stage |
| `ds_llama_frozen32_tp4_stage0_long.json` | Frozen32 later stages |
| `ds_llama_true_tp_stage0_config.json` | Original COMB true TP, ZeRO stage 0 |
| `ds_llama_ds0195_tp_compare.json` | Historical DeepSpeed TP comparison |
| `ds_llama_config.json` / `ds_deepseek_config.json` | Original model configs |

For the shared reproduction loop, enabled TensorBoard logging defaults to
`<output-dir>/tensorboard/` (DeepSpeed may add the job-name subdirectory).
An explicit `tensorboard.output_path` in the supplied config takes precedence.
Use an output directory outside the repository or the ignored `outputs/` directory.

Historical supervisors and converters remain available for inspecting and
recovering older runs. Some depend on missing historical launch scripts; see
the [project guide](../docs/project-layout.md#historical-tools).

## Original upstream workflow

The following instructions describe the original COMB model, not Frozen32.

## Prepare datasets

We expect that Comb model behaves the same as its backbone model, so the output of backbone model is used to train. Use the script `construct_data.py` to generate answers. Remember to specify the `model_name`.
```bash
cd ~/Comb
python data/construct_data.py
```

Since the output of `Deepseek-V2-Lite` is unsatisfactory, we use the anwsers of `Llama-3.1-8B-Instruct` to train `CombDeepseek-V2-Lite` (a.k.a. distillation). Use the script `distill_data.py`.
```bash
python data/distill_data.py
```

## Adjust batch size

To prevent Out of Memory (OOM) errors, the batch size of dataset should be specified. We divide the dataset into buckets based on the length of the context. This helps in efficient batching during training. So you should adjust `BUCKET_BATCH_SIZE` in `data/base.py` according to hardware constraints. For example, the default value is for training `CombLlama-11B-Instruct` with A100 80GB GPU.

## Launch

We use deepspeed to train the new parameters. `ds_llama_config.json` includes the configuration of deepspeed. We launch the training with the following command.
Remember to change directory to `training` folder first (`cd training`).
```bash
deepspeed --num_gpus=4 train_llama.py
```
