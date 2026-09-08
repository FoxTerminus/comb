#!/usr/bin/env bash
set -euo pipefail

repo="/data3/junhaohu/comb"
python="/data3/junhaohu/anaconda3/envs/comb/bin/python"
train_root="${COMB_TRUE_TP_TRAIN_ROOT:-/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0}"
tag="${COMB_CAPACITY_PREFLIGHT_TAG:-step_00150000}"
gpu_ids="${COMB_TRUE_TP_GPU_IDS:-3,5}"
tp_size="${COMB_TRUE_TP_SIZE:-2}"
output_root="${COMB_CAPACITY_PREFLIGHT_OUTPUT:-/data3/junhaohu/checkpoints/Comb_official_milestone_150000_true_tp_eval}"
parquet="${COMB_CAPACITY_PREFLIGHT_PARQUET:-/data3/junhaohu/.cache/huggingface/bucket_cache/Natural-Instructions_ba8df1a0b6c710073d57/bucket_11.parquet}"
scratch="${output_root}/capacity_preflight_runtime_${tag}"
report="${output_root}/long_context_capacity_preflight_${tag}.json"

if [[ ! "${tag}" =~ ^step_[0-9]{8}$ ]] \
    || [[ ! "${tp_size}" =~ ^[2-9][0-9]*$ ]] \
    || [[ ! "${gpu_ids}" =~ ^[0-7](,[0-7])+$ ]]; then
  echo "invalid tag, TP size, or GPU list" >&2
  exit 2
fi
IFS=',' read -ra gpu_array <<< "${gpu_ids}"
if (( ${#gpu_array[@]} != tp_size )); then
  echo "GPU count ${#gpu_array[@]} does not equal TP size ${tp_size}" >&2
  exit 2
fi
if [[ ! -d "${train_root}/${tag}" ]] || [[ ! -s "${parquet}" ]]; then
  echo "missing checkpoint or long-context parquet" >&2
  exit 2
fi
for gpu in "${gpu_array[@]}"; do
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} is not idle; refusing capacity preflight" >&2
    exit 3
  fi
done

mkdir -p "${output_root}" "${scratch}"
cd "${repo}"
CUDA_HOME="/usr/local/cuda-12.9" \
PATH="/usr/local/cuda-12.9/bin:/data3/junhaohu/anaconda3/envs/comb/bin:/usr/local/bin:/usr/bin:/bin" \
TORCH_EXTENSIONS_DIR="/data3/junhaohu/.cache/torch_extensions/py313_cu128" \
CUDA_VISIBLE_DEVICES="${gpu_ids}" \
PYTHONPATH="${repo}" \
HF_HOME="/data3/junhaohu/.cache/huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRITON_CACHE_DIR="/data3/junhaohu/.triton" \
XDG_CACHE_HOME="/data3/junhaohu/.cache" \
"${python}" -m torch.distributed.run --standalone \
  --nproc_per_node="${tp_size}" \
  training/preflight_true_tp_long_context.py \
  --config training/ds_llama_true_tp_stage0_config.json \
  --resume-root "${train_root}" \
  --resume-tag "${tag}" \
  --parquet "${parquet}" \
  --output-dir "${scratch}" \
  --output "${report}" \
  --tensor-parallel-size "${tp_size}"

test -s "${report}"
echo "long-context capacity preflight passed: ${report}"
