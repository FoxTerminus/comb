#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 STEP CHECKPOINT_ROOT OUTPUT_ROOT SUITE_MANIFEST" >&2
  exit 2
fi

eval_step="$1"
checkpoint_root="$2"
output_root="$3"
suite_manifest="$4"

if [[ ! "$eval_step" =~ ^[0-9]+$ ]]; then
  echo "STEP must be a non-negative integer: $eval_step" >&2
  exit 2
fi
if [[ ! -d "$checkpoint_root" ]]; then
  echo "checkpoint root does not exist: $checkpoint_root" >&2
  exit 2
fi
if [[ ! -f "$suite_manifest" ]]; then
  echo "suite manifest does not exist: $suite_manifest" >&2
  exit 2
fi

checkpoint_tag="$(printf 'step_%08d' "$eval_step")"
checkpoint_dir="$checkpoint_root/$checkpoint_tag"
squad_output="$output_root/squad"
heldout_output="$output_root/heldout_v1"

echo "waiting for $checkpoint_dir"
while true; do
  if [[ -d "$checkpoint_dir" ]]; then
    shard_count="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'mp_rank_*_model_states.pt' | wc -l)"
    if [[ "$shard_count" -eq 4 ]]; then
      valid=1
      for shard in "$checkpoint_dir"/mp_rank_*_model_states.pt; do
        unzip -t "$shard" >/dev/null || valid=0
      done
      if [[ "$valid" -eq 1 ]]; then
        break
      fi
    fi
  fi
  sleep 60
done
echo "checkpoint validated: $checkpoint_dir"

echo "waiting for GPUs 4-7 to be idle"
while ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
  awk -F, '$1 >= 4 && $1 <= 7 && $2 + 0 > 1024 { busy=1 } END { exit busy }'; do
  sleep 60
done

mkdir -p "$squad_output" "$heldout_output"

/data3/junhaohu/comb/training/launch_frozen32_milestone_eval.sh \
  "$eval_step" "$checkpoint_root" "$squad_output" \
  2>&1 | tee "$squad_output/eval_stdout.log"

/data3/junhaohu/comb/training/launch_frozen32_milestone_eval.sh \
  "$eval_step" "$checkpoint_root" "$heldout_output" "$suite_manifest" \
  2>&1 | tee "$heldout_output/eval_stdout.log"

echo "milestone evaluation complete: $output_root"
