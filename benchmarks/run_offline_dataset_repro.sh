#!/usr/bin/env bash
set -uo pipefail

if (( $# != 2 )); then
  echo "usage: $0 DATASET MODEL" >&2
  exit 64
fi

dataset="$1"
model="$2"
repo="/data3/junhaohu/comb"
python="/data3/junhaohu/anaconda3/envs/comb/bin/python"
result_dir="${repo}/benchmarks/results/offline/${model//\//_}"
result="${result_dir}/${dataset}.json"
log="/data3/junhaohu/checkpoints/Comb_official_offline_${dataset}_gpu4.log"
status_file="/data3/junhaohu/checkpoints/Comb_official_offline_${dataset}_gpu4.status"
tokenizer_snapshot="/data3/junhaohu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

mkdir -p "${result_dir}"
cd "${repo}/benchmarks" || exit 1
printf 'start time=%s dataset=%s model=%s\n' \
  "$(date --iso-8601=seconds)" "${dataset}" "${model}" >> "${log}"
printf 'running time=%s dataset=%s model=%s\n' \
  "$(date --iso-8601=seconds)" "${dataset}" "${model}" > "${status_file}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
COMB_TOKENIZER_PATH="${tokenizer_snapshot}" PYTHONUNBUFFERED=1 \
"${python}" -u offline_local_repro.py \
  --dataset "${dataset}" --model "${model}" >> "${log}" 2>&1
status=$?

if (( status == 0 )) && [[ -s "${result}" ]]; then
  printf 'complete time=%s status=0 result=%s bytes=%s\n' \
    "$(date --iso-8601=seconds)" "${result}" "$(stat -c %s "${result}")" \
    > "${status_file}"
  exit 0
fi

printf 'failed time=%s status=%d result_exists=%s result_bytes=%s\n' \
  "$(date --iso-8601=seconds)" "${status}" \
  "$([[ -e "${result}" ]] && echo true || echo false)" \
  "$([[ -e "${result}" ]] && stat -c %s "${result}" || echo 0)" \
  > "${status_file}"
exit "${status}"
