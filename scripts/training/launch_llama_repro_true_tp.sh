#!/usr/bin/env bash
set -euo pipefail

: "${COMB_GPU_IDS:?Set COMB_GPU_IDS to comma-separated physical GPU IDs}"
: "${COMB_TP_SIZE:?Set COMB_TP_SIZE to the number of GPU IDs}"
: "${COMB_OUTPUT_DIR:?Set COMB_OUTPUT_DIR to a new true-TP output directory}"

COMB_REPO_DIR="/data3/junhaohu/comb"
COMB_ENV_DIR="/data3/junhaohu/anaconda3/envs/comb"
COMB_TOKENIZER_DIR="/data3/junhaohu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
COMB_RESUME_ARGS=()
if [[ -n "${COMB_RESUME_ROOT:-}" ]]; then
  COMB_RESUME_ARGS+=(--resume-root "${COMB_RESUME_ROOT}")
  if [[ -n "${COMB_RESUME_TAG:-}" ]]; then
    COMB_RESUME_ARGS+=(--resume-tag "${COMB_RESUME_TAG}")
  fi
fi

IFS=',' read -r -a COMB_GPU_ARRAY <<< "${COMB_GPU_IDS}"
if [[ "${COMB_TP_SIZE}" -le 1 ]] || [[ "${#COMB_GPU_ARRAY[@]}" -ne "${COMB_TP_SIZE}" ]]; then
  echo "COMB_TP_SIZE must be >1 and equal the number of GPU IDs" >&2
  exit 2
fi
declare -A COMB_SEEN_GPUS=()
for COMB_GPU_ID in "${COMB_GPU_ARRAY[@]}"; do
  if [[ ! "${COMB_GPU_ID}" =~ ^[0-7]$ ]] || [[ -n "${COMB_SEEN_GPUS[${COMB_GPU_ID}]:-}" ]]; then
    echo "GPU IDs must be distinct integers in the range 0-7" >&2
    exit 2
  fi
  COMB_SEEN_GPUS["${COMB_GPU_ID}"]=1
  COMB_GPU_PIDS="$(nvidia-smi -i "${COMB_GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${COMB_GPU_PIDS//[[:space:]]/}" ]]; then
    echo "GPU ${COMB_GPU_ID} already has compute process(es): ${COMB_GPU_PIDS//$'\n'/, }" >&2
    exit 3
  fi
done

if [[ -e "${COMB_OUTPUT_DIR}" ]] && [[ -z "${COMB_ALLOW_EXISTING_OUTPUT:-}" ]]; then
  echo "Output already exists; set COMB_ALLOW_EXISTING_OUTPUT=1 only for an intentional resume" >&2
  exit 4
fi

export CUDA_HOME="/usr/local/cuda-12.9"
export PATH="${CUDA_HOME}/bin:${COMB_ENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${COMB_REPO_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export COMB_TOKENIZER_PATH="${COMB_TOKENIZER_DIR}"
export TRITON_CACHE_DIR="/data3/junhaohu/.triton"
export XDG_CACHE_HOME="/data3/junhaohu/.cache"
export VLLM_CACHE_ROOT="/data3/junhaohu/.cache/vllm"
COMB_DS_CONFIG="${COMB_DS_CONFIG:-ds_llama_config.json}"
if [[ "${COMB_DS_CONFIG}" == */* ]] || [[ ! -f "${COMB_REPO_DIR}/training/${COMB_DS_CONFIG}" ]]; then
  echo "COMB_DS_CONFIG must name a config file in training/: ${COMB_DS_CONFIG}" >&2
  exit 5
fi

cd "${COMB_REPO_DIR}/training"
exec deepspeed \
  --include "localhost:${COMB_GPU_IDS}" \
  --master_port "${COMB_MASTER_PORT:-29672}" \
  train_llama_true_tp_repro.py \
  --config "${COMB_DS_CONFIG}" \
  --output-dir "${COMB_OUTPUT_DIR}" \
  --max-optimizer-steps "${COMB_MAX_STEPS:-0}" \
  --log-interval "${COMB_LOG_INTERVAL:-1}" \
  --save-interval "${COMB_SAVE_INTERVAL:-1000}" \
  --keep-last-checkpoints "${COMB_KEEP_LAST_CHECKPOINTS:-2}" \
  --num-workers "${COMB_NUM_WORKERS:-4}" \
  --tensor-parallel-size "${COMB_TP_SIZE}" \
  --context-eval-examples "${COMB_CONTEXT_EVAL_EXAMPLES:-0}" \
  --no-export-final-hf \
  "${COMB_RESUME_ARGS[@]}"
