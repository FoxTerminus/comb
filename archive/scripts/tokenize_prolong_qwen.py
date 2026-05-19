#!/usr/bin/env python3
"""Convert Llama2-tokenized ProLong LITPKDS to Qwen3-tokenized LITPKDS.

Pipeline:
  Llama2 LITPKDS .bin → Llama2 tokenizer decode → text → Qwen3 tokenizer encode → new LITPKDS .bin

Usage:
  python scripts/tokenize_prolong_qwen.py \
    --input-dir /data3/junhaohu/data/prolong_64K_v2/prolong_64K_v2 \
    --output-dir /data3/junhaohu/data/prolong_qwen_v2 \
    --subset 20 \
    --start 0
"""

import argparse
import os
import struct
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# LITPKDS format constants
HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24

DTYPE_REV = {np.dtype("uint16"): 8, np.dtype("int32"): 4}


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(len(HDR_MAGIC))
        assert magic == HDR_MAGIC
        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1
        dtype_code = struct.unpack("<B", f.read(1))[0]
        chunk_size = struct.unpack("<Q", f.read(8))[0]
    return dtype_code, chunk_size


def write_litpkds(path, tokens):
    dtype = np.int32
    data = np.array(tokens, dtype=dtype).tobytes()
    with open(path, "wb") as f:
        f.write(HDR_MAGIC)
        f.write(struct.pack("<Q", 1))          # version
        f.write(struct.pack("<B", 4))          # dtype=int32
        f.write(struct.pack("<Q", len(tokens)))  # chunk_size
        f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", type=int, default=20, help="1/N files (default: 20)")
    parser.add_argument("--start", type=int, default=0, help="start index for parallel runs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    llama2_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    qwen3_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    bins = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".bin")])
    # Only train files
    bins = [f for f in bins if f.startswith("train_")]
    # Subset
    bins = bins[args.start::args.subset]

    print(f"Processing {len(bins)} files (subset=1/{args.subset}, start={args.start})")

    for fname in tqdm(bins):
        ipath = os.path.join(args.input_dir, fname)
        opath = os.path.join(args.output_dir, fname)
        if os.path.exists(opath):
            continue

        dtype_code, _ = read_header(ipath)
        dtype_map = {8: np.dtype(np.uint16), 4: np.dtype(np.int32)}
        dtype = dtype_map[dtype_code]
        num_tokens = (os.path.getsize(ipath) - HDR_SIZE) // dtype.itemsize

        arr = np.memmap(ipath, dtype=dtype, mode="r", offset=HDR_SIZE, shape=(num_tokens,))
        tokens = arr.astype(np.int64)

        # Decode Llama2 → text → Encode Qwen3 (in chunks to avoid memory issues)
        chunk_size = 50000
        new_tokens = []
        for i in range(0, len(tokens), chunk_size):
            batch = tokens[i : i + chunk_size]
            text = llama2_tok.decode(batch.tolist(), skip_special_tokens=True)
            new_tokens.extend(qwen3_tok.encode(text, add_special_tokens=False))

        write_litpkds(opath, new_tokens)

    print("Done.")


if __name__ == "__main__":
    main()
