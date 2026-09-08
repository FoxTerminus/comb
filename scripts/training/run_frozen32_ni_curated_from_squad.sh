#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data3/junhaohu/comb
export HF_HOME=/data3/junhaohu/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COMB_TOKENIZER_PATH=/data3/junhaohu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

PYTHON=/data3/junhaohu/anaconda3/envs/comb-ds0195/bin/python
TRAIN=/data3/junhaohu/comb/training/train_llama_frozen32.py
SQUAD_ROOT=/data3/junhaohu/checkpoints/CombLlamaFrozen32/squad_rebuild_for_ni_curated
NI_ROOT=/data3/junhaohu/checkpoints/CombLlamaFrozen32/ni_curated_from_squad

if [[ -e "$SQUAD_ROOT" || -e "$NI_ROOT" ]]; then
    echo "Refusing to overwrite an existing output directory" >&2
    exit 1
fi

cd /data3/junhaohu/comb/training
export FROZEN32_DATASETS=SQuAD
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 "$TRAIN" \
    --config ds_llama_frozen32_tp4_stage0.json \
    --output-dir "$SQUAD_ROOT" \
    --tensor-parallel-size 4 \
    --save-interval 1000 \
    --keep-last-checkpoints 1 \
    --context-eval-examples 64 \
    2>&1 | tee "${SQUAD_ROOT}_bootstrap.log"

export FROZEN32_DATASETS=Natural-Instructions-Curated
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 "$TRAIN" \
    --config ds_llama_frozen32_tp4_stage0_long.json \
    --output-dir "$NI_ROOT" \
    --tensor-parallel-size 4 \
    --save-interval 1000 \
    --keep-last-checkpoints 2 \
    --context-eval-examples 64 \
    --resume-root "$SQUAD_ROOT" \
    --resume-tag step_00004073 \
    --resume-reset-data-position \
    --resume-reset-lr-scheduler \
    2>&1 | tee "${NI_ROOT}_bootstrap.log"
