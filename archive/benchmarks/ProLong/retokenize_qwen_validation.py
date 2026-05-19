#!/usr/bin/env python
"""Retokenize ProLong validation files from Llama2 tokens to Qwen3 tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from litpkds import HDR_SIZE, file_num_tokens, read_header, write_litpkds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/data3/junhaohu/data/prolong_64K_v2/prolong_64K_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="/data3/junhaohu/data/prolong_qwen_v2_validation",
    )
    parser.add_argument("--llama-tokenizer", default="/data3/junhaohu/model/Llama-2-7b-hf-tokenizer")
    parser.add_argument("--qwen-tokenizer", default="/data3/junhaohu/model/Qwen3-0.6B")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llama_tok = AutoTokenizer.from_pretrained(args.llama_tokenizer)
    qwen_tok = AutoTokenizer.from_pretrained(args.qwen_tokenizer)

    files = sorted(input_dir.glob("validation*.bin"))
    files = files[args.start :: args.stride]
    if args.limit_files is not None:
        files = files[: args.limit_files]
    print(f"Retokenizing {len(files)} validation files to {output_dir}")

    for ipath in tqdm(files):
        opath = output_dir / ipath.name
        if opath.exists() and not args.overwrite:
            continue
        dtype, _ = read_header(ipath)
        num_tokens = file_num_tokens(ipath, dtype)
        arr = np.memmap(ipath, dtype=dtype, mode="r", offset=HDR_SIZE, shape=(num_tokens,))
        new_tokens: list[int] = []
        for i in range(0, num_tokens, args.chunk_size):
            ids = arr[i : i + args.chunk_size].astype(np.int64).tolist()
            text = llama_tok.decode(ids, skip_special_tokens=True)
            new_tokens.extend(qwen_tok.encode(text, add_special_tokens=False))
        write_litpkds(opath, new_tokens, dtype=np.int32)


if __name__ == "__main__":
    main()
