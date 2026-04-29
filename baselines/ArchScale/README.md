# ArchScale Baselines

This directory provides the shared implementation for paper-aligned SambaY and
Samba+YOCO baselines. The model internals are vendored from the official
Microsoft ArchScale repository under `vendor/archscale`; local code only wraps
the official configs and model constructors.

The primary d=8 targets are:

| Alias | Upstream config | Width | Layers | Q heads | KV heads | Head dim | MLP | Paper total params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sambay` | `sambay_d8` | 992 | 8 | 8 | 2 | 128 | 3968 | 155.0M |
| `sambayoco` | `sambayoco_d8` | 1008 | 8 | 8 | 2 | 128 | 4032 | 155.4M |

The paper parameter counts assume the ArchScale scaling setup with a 32K
vocabulary and untied embeddings. Overriding the tokenizer vocabulary changes
the embedding parameters but preserves the non-embedding architecture shape.

Example:

```bash
PYTHONPATH=/data3/junhaohu/comb \
python -m baselines.ArchScale.tools.inspect_model --architecture sambay
```
