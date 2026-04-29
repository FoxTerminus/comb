# SambaY Baseline

This directory contains the SambaY-Llama baseline plan and implementation.

The baseline is designed as an independent `decoder-hybrid-decoder` model for
comparison against `CombLlama` and `YOCO`. It uses the same raw training
datasets where appropriate, but uses its own preprocessing and collate path for
SambaY inputs.

Current implementation status:

1. CPU/tiny correctness for model structure, GMU, preprocessing, and training.
2. Llama initialization from the local Llama-3.1-8B-Instruct checkpoint.
3. Cache-enabled generation.
4. Tensor-parallel `torchrun` training smoke.

Full training on the same 2,3,4,7 physical GPUs used by the YOCO baseline:

```bash
cd /data3/junhaohu/comb
CUDA_VISIBLE_DEVICES=2,3,4,7 /data3/junhaohu/anaconda3/envs/mamba/bin/python \
  baselines/SambaY/training/preflight_full_tp4.py

bash baselines/SambaY/training/run_full_tp4.sh
```

Operational notes:

- The launcher refuses physical GPU0/GPU1.
- Formal TP training uses `--mamba-tp-mode replicated-official`: official
  ArchScale/Mamba kernels are replicated per TP rank, while surrounding
  projections are sharded.
- Interval checkpoints default to every 1000 optimizer steps; override with
  `SAVE_INTERVAL=...`.
- Resume from the latest interval checkpoint with
  `RESUME_CKPT=/data3/junhaohu/checkpoints/SambaY-Llama-8B/step_N`.
- Use TP shard checkpoints for resume. By default, distributed training merges
  the final shard checkpoint into `OUTPUT_DIR` after the run. To merge a
  different interval checkpoint manually:

```bash
/data3/junhaohu/anaconda3/envs/mamba/bin/python \
  baselines/SambaY/scripts/merge_tp_checkpoint.py \
  --tp-checkpoint-dir /data3/junhaohu/checkpoints/SambaY-Llama-8B/step_N \
  --base-model-path /data3/junhaohu/model/SambaY-Llama-8B-Init \
  --output-dir /data3/junhaohu/model/SambaY-Llama-8B-Merged
```

- Stop long runs after an interval checkpoint whenever possible.
- Check training health with:

```bash
/data3/junhaohu/anaconda3/envs/mamba/bin/python \
  baselines/SambaY/training/summarize_diagnostics.py \
  baselines/SambaY/training/training_diagnostics.csv
```
