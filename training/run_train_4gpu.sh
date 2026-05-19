#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="/data3/junhaohu:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
TORCHRUN="${TORCHRUN:-/data3/junhaohu/anaconda3/envs/comb/bin/torchrun}"

DATA_ARG=()
if [[ -n "${DATA:-}" ]]; then
  DATA_ARG=(--data "${DATA}")
fi

STEP_ARG=()
if [[ -n "${TOTAL_STEPS:-}" ]]; then
  STEP_ARG=(--total-steps "${TOTAL_STEPS}")
fi

"${TORCHRUN}" --standalone --nproc_per_node=4 \
  /data3/junhaohu/comb/training/train.py \
  --text-model "${TEXT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
  "${DATA_ARG[@]}" \
  "${STEP_ARG[@]}" \
  --output-dir "${OUTPUT_DIR:-/data3/junhaohu/checkpoints/CombLlama}" \
  --warmup-steps "${WARMUP_STEPS:-100}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-64}" \
  --max-text-tokens-per-batch "${MAX_TEXT_TOKENS_PER_BATCH:-8192}" \
  --max-chunk-tokens-per-batch "${MAX_CHUNK_TOKENS_PER_BATCH:-32768}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --lr "${LR:-5e-5}" \
  --lr-schedule-steps "${LR_SCHEDULE_STEPS:-8000000}" \
  --log-interval "${LOG_INTERVAL:-10}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --keep-last-n "${KEEP_LAST_N:-5}" \
  --gradient-checkpointing \
  "$@"
