"""Adapter preflight checks that do not load model weights."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.scripts.utils.config import load_run_config, repo_path
from benchmarks.scripts.utils.io import read_json, write_json


class AdapterPreflightError(RuntimeError):
    """Raised when adapter configuration is not runnable in the current environment."""


@dataclass
class AdapterDiagnostics:
    adapter: str
    model: str
    model_path: str | None
    tokenizer_path: str | None
    device: str | None
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "model": self.model,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "device": self.device,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


CUSTOM_ADAPTER_MODULES = {
    "combllama": "models.CombLlama",
    "comb_llama": "models.CombLlama",
    "yoco": "baselines.YOCO.models.YOCO",
    "sambay": "baselines.SambaY.models.SambaY",
}


def diagnose_adapter_config(config: dict[str, Any]) -> AdapterDiagnostics:
    adapter = str(config.get("adapter", config.get("name", "mock"))).lower()
    model = str(config.get("name", adapter))
    model_path = _optional_text(config.get("model_path"))
    tokenizer_path = _optional_text(config.get("tokenizer_path"))
    device = _optional_text(config.get("device"))
    errors: list[str] = []
    warnings: list[str] = []

    if adapter == "mock":
        return AdapterDiagnostics(adapter, model, model_path, tokenizer_path, device, ok=True)

    if adapter not in {"llama", "combllama", "comb_llama", "yoco", "sambay"}:
        errors.append(f"unsupported adapter: {adapter}")

    _check_local_or_remote_path("model_path", model_path, errors, required=True)
    _check_local_or_remote_path("tokenizer_path", tokenizer_path, errors, required=adapter != "mock")
    _check_package("transformers", errors)
    _check_package("torch", errors)
    _check_cuda(device, errors, warnings)

    custom_module = CUSTOM_ADAPTER_MODULES.get(adapter)
    if custom_module:
        _check_module(custom_module, errors)

    return AdapterDiagnostics(
        adapter=adapter,
        model=model,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


def raise_for_diagnostics(diagnostics: AdapterDiagnostics) -> None:
    if not diagnostics.ok:
        raise AdapterPreflightError("; ".join(diagnostics.errors))


def load_model_config(path: str | Path) -> dict[str, Any]:
    config = read_json(repo_path(path))
    if "model_config" in config:
        config = load_run_config(path).get("model", {})  # type: ignore[assignment]
    return config


def _check_package(package: str, errors: list[str]) -> None:
    if importlib.util.find_spec(package) is None:
        errors.append(f"missing Python package: {package}")


def _check_module(module: str, errors: list[str]) -> None:
    try:
        spec = importlib.util.find_spec(module)
    except Exception as exc:
        errors.append(f"cannot inspect local adapter module {module}: {type(exc).__name__}: {exc}")
        return
    if spec is None:
        errors.append(f"cannot find local adapter module: {module}")


def _check_cuda(device: str | None, errors: list[str], warnings: list[str]) -> None:
    if not device or not device.startswith("cuda"):
        return
    if importlib.util.find_spec("torch") is None:
        errors.append("cuda device requested but torch is not installed")
        return
    try:
        import torch

        if not torch.cuda.is_available():
            errors.append("cuda device requested but torch.cuda.is_available() is false")
    except Exception as exc:
        warnings.append(f"could not inspect CUDA availability: {type(exc).__name__}: {exc}")


def _check_local_or_remote_path(name: str, value: str | None, errors: list[str], *, required: bool) -> None:
    if not value:
        if required:
            errors.append(f"missing required {name}")
        return
    if not _looks_like_local_path(value):
        return
    path = repo_path(Path(value).expanduser())
    if not path.exists():
        errors.append(f"{name} does not exist: {path}")


def _looks_like_local_path(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() or value.startswith(".") or path.exists()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check adapter config without loading model weights")
    parser.add_argument("--model-config", action="append", default=[])
    parser.add_argument("--run-config", action="append", default=[])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs: list[dict[str, Any]] = []
    for path in args.model_config:
        configs.append(load_model_config(path))
    for path in args.run_config:
        configs.append(load_run_config(path).get("model", {}))
    if not configs:
        for path in sorted((repo_path("benchmarks/configs/models")).glob("*.json")):
            configs.append(read_json(path))
    results = [diagnose_adapter_config(config).to_dict() for config in configs]
    payload = {"ok": all(row["ok"] for row in results), "results": results}
    if args.output:
        write_json(args.output, payload)
    for row in results:
        status = "ok" if row["ok"] else "error"
        print(f"{status}: {row['model']} ({row['adapter']})")
        for error in row["errors"]:
            print(f"  error: {error}")
        for warning in row["warnings"]:
            print(f"  warning: {warning}")
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
