# ProLong Validation

This evaluates the three ProLong-trained models on the ProLong validation split
with each model's native training objective.

- `sambay` and `sambayoco`: Llama2-tokenized `validation*.bin`, 64K causal LM loss.
- `comb-qwen`: Qwen3-tokenized validation files, 64K chunk + 32K target loss.

The losses are useful for checking whether each model learned its own
ProLong objective. Because tokenizer and objective differ, compare `sambay`
against `sambayoco` directly; compare `comb-qwen` as its own native objective.

## Prepare Comb Validation Data

The Llama2-tokenized validation files already exist at:

```text
/data3/junhaohu/data/prolong_64K_v2/prolong_64K_v2
```

Training used the 1/20 ProLong subset with file ids `mod 20 == 0`, and the
validation files were deleted from that subset. This benchmark therefore uses
the same 1/20 validation slice from the full validation directory by default:
`validation_...0000, validation_...0020, validation_...0040, ...`.

For `comb-qwen`, retokenize that same 1/20 validation slice to Qwen3 tokens first:

```bash
cd /data3/junhaohu/comb/benchmarks/ProLong
/data3/junhaohu/anaconda3/envs/samba/bin/python retokenize_qwen_validation.py \
  --limit-files 64
```

Omit `--limit-files` to retokenize the whole 1/20 validation slice. Use
`--stride 1` only if you intentionally want the full validation split.

## Run A Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 /data3/junhaohu/anaconda3/envs/samba/bin/python eval.py \
  --model sambay \
  --max-samples 2

CUDA_VISIBLE_DEVICES=1 /data3/junhaohu/anaconda3/envs/samba/bin/python eval.py \
  --model sambayoco \
  --max-samples 2

CUDA_VISIBLE_DEVICES=2 /data3/junhaohu/anaconda3/envs/samba/bin/python eval.py \
  --model comb-qwen \
  --max-samples 2
```

## Run Larger Validation

```bash
CUDA_VISIBLE_DEVICES=0 /data3/junhaohu/anaconda3/envs/samba/bin/python eval.py --model sambay --max-samples 128
CUDA_VISIBLE_DEVICES=1 /data3/junhaohu/anaconda3/envs/samba/bin/python eval.py --model sambayoco --max-samples 128
CUDA_VISIBLE_DEVICES=2 /data3/junhaohu/anaconda3/envs/samba/bin/python eval.py --model comb-qwen --max-samples 128
```

Collect:

```bash
/data3/junhaohu/anaconda3/envs/samba/bin/python collect.py
```

## Multi-GPU Sharding

Use `launch.py` to run one process per GPU. It shards the default 1/20
validation slice by composing `--file-start/--file-stride` with
`--shard-id/--num-shards`, and writes shard-safe files such as
`sambay.shard0-of-4_summary.json`.

```bash
cd /data3/junhaohu/comb/benchmarks/ProLong

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model sambay \
  --gpus 0,1,2,3

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model sambayoco \
  --gpus 0,1,2,3

/data3/junhaohu/anaconda3/envs/samba/bin/python launch.py \
  --model comb-qwen \
  --gpus 0,1,2,3
```

Collect computes token-weighted mean loss and perplexity:

```bash
/data3/junhaohu/anaconda3/envs/samba/bin/python collect.py
```

## Plot Loss Curves

Draw per-sample validation loss curves from all `*_details.csv` files:

```bash
/data3/junhaohu/anaconda3/envs/samba/bin/python plot_loss_curve.py \
  --results-dir /data3/junhaohu/comb/benchmarks/ProLong/results/validation \
  --rolling-window 16
```

This writes:

- `validation_loss_curve.png`
- `validation_loss_curve.pdf`
- `validation_loss_curve.csv`
