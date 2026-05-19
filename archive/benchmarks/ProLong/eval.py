#!/usr/bin/env python
"""Evaluate SambaY, SambaYOCO, and Comb-Qwen on ProLong validation loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from litpkds import LitpkdsBlockDataset


ROOT = Path("/data3/junhaohu")
REPO_ROOT = ROOT / "comb"
MODEL_SPECS = {
    "comb-qwen": {
        "kind": "comb",
        "path": ROOT / "model" / "Comb-Qwen3-1B",
        "data_dir": ROOT / "data" / "prolong_qwen_v2_validation",
        "block_size": 65536 + 32768,
    },
    "sambay": {
        "kind": "samba",
        "path": ROOT / "model" / "SambaY-1B",
        "data_dir": ROOT / "data" / "prolong_64K_v2" / "prolong_64K_v2",
        "block_size": 65536,
    },
    "sambayoco": {
        "kind": "samba",
        "path": ROOT / "model" / "SambaYOCO-1B",
        "data_dir": ROOT / "data" / "prolong_64K_v2" / "prolong_64K_v2",
        "block_size": 65536,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks" / "ProLong" / "results"))
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default="validation*.bin")
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-stride", type=int, default=20)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--comb-ctx-len", type=int, default=65536)
    parser.add_argument("--comb-target-len", type=int, default=32768)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def install_local_comb_package() -> None:
    package_name = "_prolong_comb_models"
    if package_name in sys.modules:
        return
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / "models")]
    sys.modules[package_name] = package


def load_comb(path: Path, dtype: torch.dtype, device: str):
    install_local_comb_package()
    from _prolong_comb_models.comb_qwen import CombForConditionalGeneration
    from _prolong_comb_models.config import CombConfig

    config = CombConfig.from_pretrained(str(path))
    model = CombForConditionalGeneration(config, from_scratch=False)
    state = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_samba(path: Path, dtype: torch.dtype, device: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from baselines.ArchScale.models.config import Config
    from baselines.ArchScale.models.model import GPT

    with (path / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)
    model = GPT(Config(**cfg))
    state = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def make_comb_metadata(batch_size: int, q_len: int, k_len: int, device: torch.device):
    cu_q = torch.arange(0, (batch_size + 1) * q_len, q_len, device=device, dtype=torch.int32)
    cu_k = torch.arange(0, (batch_size + 1) * k_len, k_len, device=device, dtype=torch.int32)
    return cu_q, cu_k, cu_k, q_len, k_len, k_len


def eval_comb_batch(model, tokens: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, int]:
    ctx_len = args.comb_ctx_len
    target_len = args.comb_target_len
    chunk_ids = tokens[:, :ctx_len].contiguous()
    input_ids = tokens[:, ctx_len - 1 : ctx_len + target_len - 1].contiguous()
    shift_labels = tokens[:, ctx_len : ctx_len + target_len].contiguous()
    position_ids_k = torch.arange(ctx_len, device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
    position_ids = torch.arange(target_len, device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
    cu_q, cu_k, cu_chunk, max_q, max_k, max_chunk = make_comb_metadata(
        tokens.shape[0], target_len, ctx_len, tokens.device
    )
    out = model(
        input_ids=input_ids,
        chunk_ids=chunk_ids,
        shift_labels=shift_labels,
        position_ids=position_ids,
        position_ids_k=position_ids_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_seqlens_chunk=cu_chunk,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        max_seqlen_chunk=max_chunk,
    )
    ntokens = int((shift_labels != -100).sum().item())
    return out.loss, ntokens


def eval_samba_batch(model, tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    out = model(input_ids=tokens, labels=tokens)
    ntokens = int(tokens[:, 1:].numel())
    return out["loss"], ntokens


def main() -> None:
    args = parse_args()
    if (args.shard_id is None) ^ (args.num_shards is None):
        raise ValueError("--shard-id and --num-shards must be provided together")
    if args.shard_id is not None:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError("--shard-id must be in [0, --num-shards)")
        args.file_start = args.file_start + args.file_stride * args.shard_id
        args.file_stride = args.file_stride * args.num_shards

    spec = MODEL_SPECS[args.model]
    dtype = torch_dtype(args.dtype)
    data_dir = Path(args.data_dir) if args.data_dir else spec["data_dir"]
    block_size = args.comb_ctx_len + args.comb_target_len if spec["kind"] == "comb" else spec["block_size"]

    dataset = LitpkdsBlockDataset(
        data_dir=data_dir,
        block_size=block_size,
        pattern=args.pattern,
        seed=args.seed,
        max_samples=args.max_samples,
        file_start=args.file_start,
        file_stride=args.file_stride,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"model={args.model} samples={len(dataset)} data_dir={data_dir} block_size={block_size}", flush=True)

    model = load_comb(spec["path"], dtype, args.device) if spec["kind"] == "comb" else load_samba(spec["path"], dtype, args.device)
    total_loss = 0.0
    total_tokens = 0
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            tokens = batch["input_ids"].to(args.device, non_blocking=True)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=args.device.startswith("cuda") and dtype != torch.float32):
                loss, ntokens = eval_comb_batch(model, tokens, args) if spec["kind"] == "comb" else eval_samba_batch(model, tokens)
            loss_value = float(loss.detach().cpu())
            elapsed = time.perf_counter() - started
            total_loss += loss_value * ntokens
            total_tokens += ntokens
            row = {
                "model": args.model,
                "sample_index": idx,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "loss": loss_value,
                "tokens": ntokens,
                "ppl": math.exp(min(loss_value, 20.0)),
                "latency_sec": elapsed,
            }
            rows.append(row)
            print(
                f"[{idx + 1}/{len(loader)}] loss={loss_value:.4f} "
                f"ppl={row['ppl']:.2f} tokens={ntokens} latency={elapsed:.2f}s",
                flush=True,
            )

    mean_loss = total_loss / max(total_tokens, 1)
    summary = {
        "model": args.model,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_samples": len(rows),
        "tokens": total_tokens,
        "loss": mean_loss,
        "ppl": math.exp(min(mean_loss, 20.0)),
        "elapsed_sec": time.perf_counter() - started_all,
        "data_dir": str(data_dir),
        "block_size": block_size,
        "file_start": args.file_start,
        "file_stride": args.file_stride,
        "note": "Native validation objective; tokenizer/objective differs between comb-qwen and samba models.",
    }

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{args.model}.shard{args.shard_id}-of-{args.num_shards}"
        if args.shard_id is not None
        else args.model
    )
    detail_path = out_dir / f"{suffix}_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    summary_path = out_dir / f"{suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
