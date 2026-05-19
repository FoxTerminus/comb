# LoCoMo Benchmark

This benchmark evaluates four model interfaces:

- `comb-qwen`: `/data3/junhaohu/model/Comb-Qwen3-1B`, trained on `prolong-64k`
- `sambayoco`: `/data3/junhaohu/model/SambaYOCO-1B`, trained on `prolong-64k`
- `sambay`: `/data3/junhaohu/model/SambaY-1B`, trained on `prolong-64k`

## Comb Chunk/Input Split

For `comb-qwen`, the complete multi-session conversation is encoded as
`chunk_ids`. The per-question instruction and question are encoded as
`input_ids`. The question is never placed in the chunk, and the full
conversation is not repeated in the input.

For `sambayoco`, and `sambay`, the input is the full
conversation followed by the question prompt.

## Multi-GPU Sharding

One process runs one model shard on one GPU. Shards are assigned by
`global_index % num_shards == shard_id`.

Example:

```bash
cd /data3/junhaohu/comb/benchmarks/LoCoMo
/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model comb-qwen \
  --gpus 0,1,2,3 \
  --python /data3/junhaohu/anaconda3/envs/samba/bin/python \
  --repetition-penalty 1.15
```

Use the `samba` environment for all full benchmark runs. It has the
Mamba/causal-conv runtime needed by `sambay` and `sambayoco`, and also runs
`comb-qwen`.

Single shard:

```bash
CUDA_VISIBLE_DEVICES=0 /data3/junhaohu/anaconda3/envs/samba/bin/python run.py \
  --model comb-qwen \
  --shard-id 0 \
  --num-shards 4 \
  --repetition-penalty 1.15
```

Smoke test:

```bash
/data3/junhaohu/anaconda3/envs/samba/bin/python run.py \
  --model comb-qwen \
  --limit 2 \
  --device cuda
```

## Scoring And Plots

After all shards finish:

```bash
python report.py --run-name full
python plot.py --run-name full
```

Outputs:

- `results/full/predictions/*.jsonl`
- `results/full/scored_predictions.json`
- `results/full/scores.csv`
- `results/full/scores.md`
- `results/full/plots/overall_scores.{png,pdf}`
- `results/full/plots/category_scores.{png,pdf}`
- `results/full/plots/conversation_heatmap.{png,pdf}`

Scoring follows the official LoCoMo QA logic:

- category 1: split multi-answer F1
- categories 2, 3, 4: stemmed token F1
- category 5: `No information available` / `not mentioned` accuracy
