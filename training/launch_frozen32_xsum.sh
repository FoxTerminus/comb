#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data3/junhaohu/comb
export HF_HOME=/data3/junhaohu/.cache/huggingface
export FROZEN32_DATASETS=XSum

exec /data3/junhaohu/anaconda3/envs/comb-ds0195/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_llama_frozen32.py \
  --config ds_llama_frozen32_tp4_stage0_long.json \
  --output-dir /data3/junhaohu/checkpoints/CombLlamaFrozen32/xsum_stage3 \
  --tensor-parallel-size 4 \
  --max-optimizer-steps 203068 \
  --save-interval 500 \
  --keep-last-checkpoints 2 \
  --context-eval-examples 64 \
  --context-eval-batch-size 4 \
  --num-workers 4 \
  --resume-root /data3/junhaohu/checkpoints/CombLlamaFrozen32/ni_stage2 \
  --resume-tag step_00196690 \
  --resume-reset-data-position
