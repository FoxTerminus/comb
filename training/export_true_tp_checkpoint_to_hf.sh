#!/usr/bin/env bash
set -euo pipefail

: "${COMB_GPU_IDS:?Set COMB_GPU_IDS to comma-separated physical GPU IDs}"
: "${COMB_TP_SIZE:?Set COMB_TP_SIZE to the number of GPU IDs}"
: "${COMB_RESUME_ROOT:?Set COMB_RESUME_ROOT to the checkpoint root}"
: "${COMB_RESUME_TAG:?Set COMB_RESUME_TAG to the checkpoint tag}"
: "${COMB_OUTPUT_DIR:?Set COMB_OUTPUT_DIR to the HF export work directory}"

repo="/data3/junhaohu/comb"
env_dir="/data3/junhaohu/anaconda3/envs/comb"
tokenizer="/data3/junhaohu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
ds_config="${COMB_DS_CONFIG:-ds_llama_true_tp_stage0_config.json}"
if [[ "${ds_config}" == */* ]] || [[ ! -f "${repo}/training/${ds_config}" ]]; then
  echo "COMB_DS_CONFIG must name a config file in training/: ${ds_config}" >&2
  exit 2
fi

IFS=',' read -ra gpu_array <<< "${COMB_GPU_IDS}"
if [[ ! "${COMB_TP_SIZE}" =~ ^[2-9][0-9]*$ ]] \
    || (( ${#gpu_array[@]} != COMB_TP_SIZE )); then
  echo "COMB_TP_SIZE must be >=2 and equal the number of GPU IDs" >&2
  exit 2
fi
declare -A seen=()
for gpu in "${gpu_array[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-7]$ ]] || [[ -n "${seen[${gpu}]:-}" ]]; then
    echo "GPU IDs must be distinct integers in the range 0-7" >&2
    exit 2
  fi
  seen["${gpu}"]=1
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} is busy: ${pids//$'\n'/, }" >&2
    exit 3
  fi
done

if [[ ! -d "${COMB_RESUME_ROOT}/${COMB_RESUME_TAG}" ]]; then
  echo "checkpoint does not exist: ${COMB_RESUME_ROOT}/${COMB_RESUME_TAG}" >&2
  exit 4
fi
hf_dir="${COMB_OUTPUT_DIR}/hf_${COMB_RESUME_TAG}"
if [[ -e "${hf_dir}" ]]; then
  echo "refusing to overwrite HF export: ${hf_dir}" >&2
  exit 5
fi
mkdir -p "${COMB_OUTPUT_DIR}"

export CUDA_HOME="/usr/local/cuda-12.9"
export PATH="${CUDA_HOME}/bin:${env_dir}/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${repo}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export COMB_TOKENIZER_PATH="${tokenizer}"
export TRITON_CACHE_DIR="/data3/junhaohu/.triton"
export XDG_CACHE_HOME="/data3/junhaohu/.cache"
export VLLM_CACHE_ROOT="/data3/junhaohu/.cache/vllm"

cd "${repo}/training"
exec "${env_dir}/bin/deepspeed" \
  --include "localhost:${COMB_GPU_IDS}" \
  --master_port "${COMB_MASTER_PORT:-29691}" \
  train_llama_true_tp_repro.py \
  --config "${ds_config}" --output-dir "${COMB_OUTPUT_DIR}" \
  --tensor-parallel-size "${COMB_TP_SIZE}" \
  --resume-root "${COMB_RESUME_ROOT}" --resume-tag "${COMB_RESUME_TAG}" \
  --export-only --no-save-final --context-eval-examples 0
