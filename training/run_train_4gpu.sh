#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="/data3/junhaohu:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
TORCHRUN="${TORCHRUN:-/data3/junhaohu/anaconda3/envs/comb/bin/torchrun}"
DEFAULT_RESUME=""
DEFAULT_INIT_FROM=""
DEFAULT_INIT_STEP="0"

DATA_ARG=()
if [[ -n "${DATA:-}" ]]; then
  DATA_ARG=(--data "${DATA}")
fi

STEP_ARG=()
if [[ -n "${TOTAL_STEPS:-}" ]]; then
  STEP_ARG=(--total-steps "${TOTAL_STEPS}")
fi

STEP_OFFSET_ARG=()
if [[ -n "${STEP_OFFSET:-}" ]]; then
  STEP_OFFSET_ARG=(--step-offset "${STEP_OFFSET}")
fi

LOAD_ARG=()
RESUME_PATH="${RESUME:-${DEFAULT_RESUME}}"
if [[ -n "${RESUME_PATH}" ]]; then
  LOAD_ARG=(--resume "${RESUME_PATH}")
elif [[ -n "${INIT_FROM:-${DEFAULT_INIT_FROM}}" ]]; then
  LOAD_ARG=(--init-from "${INIT_FROM:-${DEFAULT_INIT_FROM}}" --init-step "${INIT_STEP:-${DEFAULT_INIT_STEP}}")
fi

"${TORCHRUN}" --standalone --nproc_per_node=4 \
  /data3/junhaohu/comb/training/train.py \
  --text-model "${TEXT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
  "${DATA_ARG[@]}" \
  "${STEP_ARG[@]}" \
  "${STEP_OFFSET_ARG[@]}" \
  "${LOAD_ARG[@]}" \
  --output-dir "${OUTPUT_DIR:-/data3/junhaohu/checkpoints/CombLlama_e32}" \
  --warmup-steps "${WARMUP_STEPS:-100}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-32}" \
  --max-text-len "${MAX_TEXT_LEN:-2048}" \
  --max-text-tokens-per-batch "${MAX_TEXT_TOKENS_PER_BATCH:-8192}" \
  --max-chunk-len "${MAX_CHUNK_LEN:-65536}" \
  --max-chunk-tokens-per-batch "${MAX_CHUNK_TOKENS_PER_BATCH:-49152}" \
  --short-chunk-subsample-len "${SHORT_CHUNK_SUBSAMPLE_LEN:-512}" \
  --short-chunk-keep-ratio "${SHORT_CHUNK_KEEP_RATIO:-0.1}" \
  --length-bucket-size "${LENGTH_BUCKET_SIZE:-4096}" \
  --grad-accum "${GRAD_ACCUM:-2}" \
  --grad-clip "${GRAD_CLIP:-0}" \
  --lr "${LR:-1e-4}" \
  --lr-schedule-steps "${LR_SCHEDULE_STEPS:-8000000}" \
  --min-lr-mult "${MIN_LR_MULT:-0.1}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --adam-beta1 "${ADAM_BETA1:-0.9}" \
  --adam-beta2 "${ADAM_BETA2:-0.95}" \
  --adam-eps "${ADAM_EPS:-1e-8}" \
  --log-interval "${LOG_INTERVAL:-10}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --keep-last-n "${KEEP_LAST_N:-3}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" \
  --eval-max-batches "${EVAL_MAX_BATCHES:-0}" \
  --validation-seed "${VALIDATION_SEED:-42}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --bf16 \
  --gradient-checkpointing \
  "$@"
