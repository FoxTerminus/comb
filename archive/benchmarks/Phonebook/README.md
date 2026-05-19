# Phonebook Benchmark

This is a synthetic long-context key-value retrieval benchmark modeled after
the Phonebook setup described in the SambaY paper: 32K context length with
about 1,850 name-number pairs and minimal instructions.

The benchmark supports the four standard interfaces:

- `comb-qwen`: trained on `prolong-64k`
- `sambayoco`: trained on `prolong-64k`
- `sambay`: trained on `prolong-64k`
- `qwen3-0.6b`: base Qwen3-0.6B, not trained on `prolong-64k`

For `comb-qwen`, the phonebook entries are fed as `chunk_ids`, and the question
is fed as the decoder `input_ids`. For the other models, the phonebook and
question are concatenated into one prompt.

## Smoke Test

```bash
cd /data3/junhaohu/comb/benchmarks/Phonebook
CUDA_VISIBLE_DEVICES=0 /data3/junhaohu/anaconda3/envs/samba/bin/python run.py \
  --model comb-qwen \
  --num-samples 2 \
  --num-pairs 128 \
  --run-name smoke \
  --overwrite
```

## Paper-Style 32K Run

The default uses `--num-pairs 1850`, matching the paper's 32K Phonebook scale.

```bash
cd /data3/junhaohu/comb/benchmarks/Phonebook

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model comb-qwen \
  --gpus 0,1,2,3 \
  --num-samples 200 \
  --num-pairs 1850 \
  --overwrite

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model sambay \
  --gpus 0,1,2,3 \
  --num-samples 200 \
  --num-pairs 1850 \
  --overwrite

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model sambayoco \
  --gpus 0,1,2,3 \
  --num-samples 200 \
  --num-pairs 1850 \
  --overwrite

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model qwen3-0.6b \
  --gpus 0,1,2,3 \
  --num-samples 200 \
  --num-pairs 1850 \
  --overwrite
```

## Score And Plot

```bash
/data3/junhaohu/anaconda3/envs/samba/bin/python report.py \
  --results-dir /data3/junhaohu/comb/benchmarks/Phonebook/results/phonebook_32k

/data3/junhaohu/anaconda3/envs/samba/bin/python plot.py \
  --results-dir /data3/junhaohu/comb/benchmarks/Phonebook/results/phonebook_32k
```

Outputs:

- `predictions/*.jsonl`: raw predictions
- `summary.csv` and `summary.md`: exact-match and contains-match scores
- `position_buckets.csv` and `position_buckets.md`: early/middle/late target position scores
- `accuracy.png/pdf`
- `position_buckets.png/pdf`
