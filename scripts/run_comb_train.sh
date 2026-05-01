#!/bin/bash
# Comb + Qwen3-0.6B formal training
set -e
export CUDA_VISIBLE_DEVICES=2,3,4,7
export PATH="/data3/junhaohu/anaconda3/envs/samba/bin:$PATH"
export PYTHONPATH="/data3/junhaohu/comb:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc_per_node=4 --master-port=29600 \
  training/train.py \
  --config /data3/junhaohu/comb/configs/comb_qwen_1b.yaml \
  --data-dir /data3/junhaohu/data/prolong_qwen_v2 \
  --ctx-len 65536 \
  --target-len 32768 \
  --total-steps 5913 \
  --warmup-steps 591 \
  --lr 3e-4 \
  --act-ckpt \
  --log-interval 10 \
  --save-interval 500 \
  --seed 42 \
  --output-dir /data3/junhaohu/checkpoints/Comb-Qwen3-1B-Prolong
