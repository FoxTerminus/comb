#!/usr/bin/env python3
"""Precompute the official post-Natural-Instructions bucket caches safely."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq

from data import DATASET_DICT
from data.base import CACHE_DIR
from training.artifact_io import write_text_atomic


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASETS = ("XSum", "Super-Natural-Instructions")
SUPPORTED_DATASETS = (
    *DEFAULT_DATASETS,
    "Natural-Instructions-Curated",
    "Super-Natural-Instructions-Curated",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_cache(wrapped, bucket_files: list[tuple[int, str]]) -> dict:
    identity = wrapped._bucket_cache_identity()
    if identity is None:
        raise RuntimeError(f"{wrapped.name} has no stable cache identity")
    identity_text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(identity_text.encode()).hexdigest()[:20]
    cache_dir = Path(CACHE_DIR) / f"{wrapped.name.replace('/', '_').replace(' ', '_')}_{cache_key}"
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"missing cache manifest: {manifest}")
    payload = json.loads(manifest.read_text())
    if payload.get("identity") != identity:
        raise RuntimeError(f"cache identity mismatch: {manifest}")
    records = []
    cached_rows = 0
    for batch_size, value in bucket_files:
        path = Path(value).resolve()
        if path.parent != cache_dir.resolve() or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe or missing cache file: {path}")
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        cached_rows += rows
        records.append(
            {
                "path": str(path),
                "rows": rows,
                "batch_size": int(batch_size),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    expected_rows = sum(
        1 for value in wrapped.data["token_count"] if 0 < int(value) <= 16_384
    )
    if cached_rows != expected_rows:
        raise RuntimeError(
            f"cached row mismatch for {wrapped.name}: {cached_rows} != {expected_rows}"
        )
    return {
        "dataset": wrapped.name,
        "input_rows": len(wrapped.data),
        "cached_rows": cached_rows,
        "excluded_rows": len(wrapped.data) - cached_rows,
        "cache_dir": str(cache_dir.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--minimum-free-gib", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unknown = [name for name in args.datasets if name not in SUPPORTED_DATASETS]
    if unknown:
        raise ValueError(f"unsupported precompute datasets: {unknown}")

    cache_root = Path(CACHE_DIR).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    lock = (cache_root / "precompute_remaining_bucket_caches.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("another remaining-cache precompute is active") from error

    result: dict[str, object] = {
        "schema_version": 1,
        "model": MODEL_NAME,
        "datasets": args.datasets,
        "started_at_unix": time.time(),
        "minimum_free_gib": args.minimum_free_gib,
        "records": [],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        free_before = shutil.disk_usage(cache_root).free / 1024**3
        if free_before < args.minimum_free_gib:
            raise RuntimeError(
                f"free disk {free_before:.2f} GiB is below floor before {name}"
            )
        wrapped = DATASET_DICT[name](MODEL_NAME, split="train")
        bucket_files = wrapped.bucketing(local_rank=0, world_size=1)
        record = verify_cache(wrapped, bucket_files)
        record["free_gib_before"] = free_before
        record["free_gib_after"] = shutil.disk_usage(cache_root).free / 1024**3
        result["records"].append(record)
        write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")

    result["finished_at_unix"] = time.time()
    result["passed"] = True
    write_text_atomic(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
