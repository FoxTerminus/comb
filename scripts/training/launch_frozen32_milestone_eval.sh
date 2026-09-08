#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 STEP CHECKPOINT_ROOT OUTPUT_DIR [SUITE_MANIFEST]" >&2
  exit 2
fi

eval_step="$1"
checkpoint_root="$2"
eval_output_dir="$3"
suite_manifest="${4:-}"

if [[ ! "$eval_step" =~ ^[0-9]+$ ]]; then
  echo "STEP must be a non-negative integer: $eval_step" >&2
  exit 2
fi
if [[ ! -d "$checkpoint_root" ]]; then
  echo "checkpoint root does not exist: $checkpoint_root" >&2
  exit 2
fi
if [[ -n "$suite_manifest" && ! -f "$suite_manifest" ]]; then
  echo "suite manifest does not exist: $suite_manifest" >&2
  exit 2
fi

checkpoint_tag="$(printf 'step_%08d' "$eval_step")"
checkpoint_dir="$checkpoint_root/$checkpoint_tag"
if [[ ! -d "$checkpoint_dir" ]]; then
  echo "checkpoint does not exist: $checkpoint_dir" >&2
  exit 2
fi

shard_count="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'mp_rank_*_model_states.pt' | wc -l)"
if [[ "$shard_count" -ne 4 ]]; then
  echo "expected four TP checkpoint shards, found $shard_count in $checkpoint_dir" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${FROZEN32_EVAL_GPUS:-4,5,6,7}"
export PYTHONPATH=/data3/junhaohu/comb
export HF_HOME=/data3/junhaohu/.cache/huggingface
export FROZEN32_DATASETS=Natural-Instructions

mkdir -p "$eval_output_dir"
cd /data3/junhaohu/comb/training

eval_args=(
  --config ds_llama_frozen32_tp4_stage0_long.json
  --output-dir "$eval_output_dir"
  --tensor-parallel-size 4
  --resume-root "$checkpoint_root"
  --resume-tag "$checkpoint_tag"
  --eval-only
  --no-save-final
  --context-eval-examples 64
  --context-eval-batch-size 4
)
if [[ -n "$suite_manifest" ]]; then
  eval_args+=(--context-eval-suite-manifest "$suite_manifest")
fi

exec /data3/junhaohu/anaconda3/envs/comb-ds0195/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_llama_frozen32.py \
  "${eval_args[@]}"
