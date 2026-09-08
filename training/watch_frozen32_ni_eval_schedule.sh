#!/usr/bin/env bash
set -euo pipefail

repo_root=/data3/junhaohu/comb
checkpoint_root=/data3/junhaohu/checkpoints/CombLlamaFrozen32/ni_stage2
eval_root=/data3/junhaohu/checkpoints/CombLlamaFrozen32
suite_manifest="$eval_root/context_eval_suite_v1/suite_manifest.json"

# The already-running 80k watcher owns that milestone.  Waiting for its tmux
# session prevents duplicate evaluation if this schedule is restarted now.
while tmux has-session -t frozen32_eval_80k 2>/dev/null; do
  sleep 60
done

cd "$repo_root"
for step in \
  90000 100000 110000 120000 130000 140000 \
  150000 160000 170000 180000 190000 196690
do
  checkpoint_tag="$(printf 'step_%08d' "$step")"
  milestone_root="$eval_root/eval_ni_step${step}"
  squad_result="$milestone_root/squad/context_dependency_${checkpoint_tag}.json"
  heldout_result="$milestone_root/heldout_v1/context_dependency_suite_${checkpoint_tag}.json"
  if [[ -s "$squad_result" && -s "$heldout_result" ]]; then
    echo "milestone already complete, skipping: $step"
    continue
  fi
  bash training/watch_frozen32_milestone_eval.sh \
    "$step" \
    "$checkpoint_root" \
    "$milestone_root" \
    "$suite_manifest"
done
