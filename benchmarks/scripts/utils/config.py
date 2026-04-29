"""Run configuration loading and resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.scripts.utils.io import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_config(path: str | Path) -> dict[str, Any]:
    config_path = repo_path(path)
    config = read_json(config_path)
    model_ref = config.get("model_config")
    if model_ref:
        model_config = read_json(repo_path(model_ref))
        config["model"] = deep_merge(model_config, config.get("model", {}))
    config.setdefault("seed", 1234)
    config.setdefault("split", "smoke")
    config.setdefault("benchmarks", ["RULER", "SCBench", "LongBench", "LoCoMo", "LongCodeBench"])
    config.setdefault("generation", {})
    config["generation"].setdefault("temperature", 0.0)
    config["generation"].setdefault("top_p", 1.0)
    config["generation"].setdefault("do_sample", False)
    config["generation"].setdefault("max_new_tokens", 32)
    config.setdefault("paths", {})
    return config

