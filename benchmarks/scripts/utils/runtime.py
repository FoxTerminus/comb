"""Runtime helpers for reproducible benchmark runs."""

from __future__ import annotations

import importlib.metadata
import os
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def make_run_id(prefix: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{prefix}_{timestamp}_{suffix}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(
    *,
    model_config: dict[str, Any],
    start_time: str,
    end_time: str | None = None,
) -> dict[str, Any]:
    status = _git(["status", "--short"])
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    except Exception:
        torch_version = None
        cuda_available = False
        cuda_device_count = 0

    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_short": status,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "torch_version": torch_version,
        "transformers_version": _version("transformers"),
        "flash_attn_version": _version("flash-attn"),
        "hostname": socket.gethostname(),
        "model_path": model_config.get("model_path"),
        "tokenizer_path": model_config.get("tokenizer_path"),
        "start_time": start_time,
        "end_time": end_time,
    }

