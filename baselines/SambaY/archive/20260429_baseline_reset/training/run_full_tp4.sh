#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data3/junhaohu/comb}"
PYTHON_ENV="${PYTHON_ENV:-/data3/junhaohu/anaconda3/envs/mamba}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,7}"

INIT_SAMBAY_PATH="${INIT_SAMBAY_PATH:-/data3/junhaohu/model/SambaY-Llama-8B-Init}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/junhaohu/model/SambaY-Llama-8B}"
CKPT_DIR="${CKPT_DIR:-/data3/junhaohu/checkpoints/SambaY-Llama-8B}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/baselines/SambaY/training}"
RESUME_ARGS=()

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
for gpu_id in "${VISIBLE_GPU_ARRAY[@]}"; do
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ "${gpu_id}" == "0" || "${gpu_id}" == "1" ]]; then
    echo "Refusing to run on forbidden physical GPU0/GPU1: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
done

if [[ -n "${RESUME_CKPT:-}" ]]; then
  RESUME_ARGS=(--resume-ckpt "${RESUME_CKPT}")
fi

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_ENV}/bin/torchrun" \
  --standalone \
  --nproc_per_node=4 \
  baselines/SambaY/training/train_sambay_megatron.py \
  --init-sambay-path "${INIT_SAMBAY_PATH}" \
  --tp-size 4 \
  --mamba-tp-mode replicated-official \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --lr "${LR:-5e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --warmup-steps "${WARMUP_STEPS:-100}" \
  --total-steps "${TOTAL_STEPS:-8000000}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --max-train-seq-len "${MAX_TRAIN_SEQ_LEN:-8192}" \
  --output-dir "${OUTPUT_DIR}" \
  --ckpt-dir "${CKPT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --steps-per-print "${STEPS_PER_PRINT:-10}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  "${RESUME_ARGS[@]}" \
  "$@"
