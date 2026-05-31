#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

PYTHON="${PYTHON:-/data3/junhaohu/anaconda3/envs/comb/bin/python}"

MODEL_TYPE="${MODEL_TYPE:-comb}"
CHECKPOINT="${CHECKPOINT:-/data3/junhaohu/checkpoints/CombLlama_Long/step_24826}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
DATASETS="${DATASETS:-hotpotqa,2wikimqa,musique,multi_news,samsum}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LIMIT="${LIMIT:-0}"
QA_MAX_NEW_TOKENS="${QA_MAX_NEW_TOKENS:-128}"
SUMMARY_MAX_NEW_TOKENS="${SUMMARY_MAX_NEW_TOKENS:-4096}"

if [[ -z "${OUTPUT_DIR}" && -n "${OUTPUT_PREFIX}" ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/results/${OUTPUT_PREFIX}"
fi

args=(
  "${SCRIPT_DIR}/LongBench.py"
  --model-type "${MODEL_TYPE}"
  --model-name "${MODEL_NAME}"
  --datasets "${DATASETS}"
  --limit "${LIMIT}"
  --qa-max-new-tokens "${QA_MAX_NEW_TOKENS}"
  --summary-max-new-tokens "${SUMMARY_MAX_NEW_TOKENS}"
)

if [[ "${MODEL_TYPE}" == "comb" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${OUTPUT_PREFIX}" ]]; then
  args+=(--output-prefix "${OUTPUT_PREFIX}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi

exec "${PYTHON}" "${args[@]}"
