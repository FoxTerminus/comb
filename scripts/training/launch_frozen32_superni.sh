#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../../training"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data3/junhaohu/comb
export HF_HOME=/data3/junhaohu/.cache/huggingface
export FROZEN32_DATASETS=Super-Natural-Instructions

exec /data3/junhaohu/anaconda3/envs/comb-ds0195/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_llama_frozen32.py \
  --config ds_llama_frozen32_tp4_stage0_long.json \
  --output-dir /data3/junhaohu/checkpoints/CombLlamaFrozen32/superni_stage4 \
  --tensor-parallel-size 4 \
  --max-optimizer-steps 265285 \
  --save-interval 1000 \
  --keep-last-checkpoints 2 \
  --context-eval-examples 64 \
  --context-eval-batch-size 4 \
  --num-workers 4 \
  --resume-root /data3/junhaohu/checkpoints/CombLlamaFrozen32/xsum_stage3 \
  --resume-tag step_00203068 \
  --resume-reset-data-position
