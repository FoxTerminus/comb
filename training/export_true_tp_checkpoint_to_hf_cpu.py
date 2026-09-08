"""Merge a native DeepSpeed AutoTP checkpoint into a Hugging Face checkpoint.

This is a CPU-only counterpart to the distributed DeepSpeed exporter.  It is
intended for evaluation snapshots when two idle GPUs are unavailable.  The
merge layout is read from the universal-checkpoint metadata stored by the
true-TP trainer; no parameter is silently guessed from its tensor shape.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from deepspeed.checkpoint.constants import UNIVERSAL_CHECKPOINT_INFO
from safetensors import safe_open
from safetensors.torch import save_file
import torch


EXPECTED_UNDECLARED_FULL_REPLICAS = {
    "chunk_model.embed_tokens.weight",
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("template_hf", type=Path)
    parser.add_argument("output_hf", type=Path)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--expected-parameters", type=int, required=True)
    return parser.parse_args()


def _matches(patterns: list[re.Pattern[str]], name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in patterns)


def load_rank_states(
    checkpoint_dir: Path, tp_size: int
) -> tuple[list[dict[str, torch.Tensor]], dict]:
    if tp_size < 2:
        raise ValueError("CPU true-TP export requires tp_size >= 2")
    states: list[dict[str, torch.Tensor]] = []
    reference_info = None
    reference_step = None
    for rank in range(tp_size):
        path = checkpoint_dir / f"mp_rank_{rank:02d}_model_states.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if int(payload.get("mp_world_size", -1)) != tp_size:
            raise RuntimeError(f"{path.name}: mp_world_size mismatch")
        step = int(payload.get("optimizer_step", payload.get("global_steps", -1)))
        if reference_step is None:
            reference_step = step
        elif step != reference_step:
            raise RuntimeError("TP ranks have different optimizer steps")
        info = payload.get(UNIVERSAL_CHECKPOINT_INFO)
        if not isinstance(info, dict):
            raise RuntimeError(f"{path.name}: missing universal checkpoint metadata")
        if reference_info is None:
            reference_info = info
        elif info != reference_info:
            raise RuntimeError("TP ranks have different universal checkpoint metadata")
        module = payload.get("module")
        if not isinstance(module, dict) or not module:
            raise RuntimeError(f"{path.name}: missing module state")
        states.append(module)
    assert reference_info is not None
    keys = set(states[0])
    for rank, state in enumerate(states[1:], 1):
        if set(state) != keys:
            raise RuntimeError(f"TP rank {rank} has a different module key set")
    return states, reference_info


def merge_tensor(
    name: str,
    tensors: list[torch.Tensor],
    replicated_patterns: list[re.Pattern[str]],
    row_patterns: list[re.Pattern[str]],
) -> tuple[torch.Tensor, str]:
    first = tensors[0]
    if any(tensor.dtype != first.dtype for tensor in tensors[1:]):
        raise RuntimeError(f"{name}: dtype mismatch across TP ranks")
    same_shape = all(tensor.shape == first.shape for tensor in tensors[1:])
    identical = same_shape and all(torch.equal(first, tensor) for tensor in tensors[1:])
    declared_replicated = _matches(replicated_patterns, name)
    if declared_replicated:
        if not identical:
            raise RuntimeError(f"{name}: declared replicated but rank values differ")
        return first, "replicated"
    # DeepSpeed leaves full vocabulary embeddings and tied/untied LM heads on
    # every rank.  No other undeclared identical TP shard is accepted.
    if identical:
        if name not in EXPECTED_UNDECLARED_FULL_REPLICAS:
            raise RuntimeError(f"{name}: unexpected identical TP rank tensors")
        return first, "identical_full"
    if first.ndim == 0:
        raise RuntimeError(f"{name}: non-identical scalar cannot be TP-sharded")
    axis = 1 if _matches(row_patterns, name) else 0
    if first.ndim <= axis:
        raise RuntimeError(f"{name}: cannot concatenate rank tensors on axis {axis}")
    for tensor in tensors[1:]:
        if tensor.ndim != first.ndim:
            raise RuntimeError(f"{name}: rank mismatch")
        for dimension, (left, right) in enumerate(zip(first.shape, tensor.shape)):
            if dimension != axis and left != right:
                raise RuntimeError(f"{name}: incompatible TP shard shapes")
    return torch.cat(tensors, dim=axis), f"concat_dim_{axis}"


def export_checkpoint(
    checkpoint_dir: Path,
    template: Path,
    output: Path,
    tp_size: int,
    expected_parameters: int,
) -> dict:
    checkpoint_dir = checkpoint_dir.resolve()
    template = template.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    index_path = template / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    states, info = load_rank_states(checkpoint_dir, tp_size)
    if set(states[0]) != set(weight_map):
        missing = sorted(set(weight_map) - set(states[0]))
        extra = sorted(set(states[0]) - set(weight_map))
        raise RuntimeError(
            f"HF/template key mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    replicated_patterns = [
        re.compile(pattern)
        for pattern in info.get("tp_replicated_parameter_patterns", [])
    ]
    row_patterns = [
        re.compile(pattern)
        for pattern in info.get("parameter_with_row_parallelism_patterns", [])
    ]
    if not replicated_patterns or not row_patterns:
        raise RuntimeError("universal checkpoint metadata lacks TP merge patterns")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    counts = {"replicated": 0, "identical_full": 0, "concat_dim_0": 0, "concat_dim_1": 0}
    identical_full_names: set[str] = set()
    parameters = 0
    try:
        model_files = set(weight_map.values())
        for source in template.iterdir():
            if source.name in model_files or source.name == index_path.name:
                continue
            if source.is_file():
                shutil.copy2(source, temporary / source.name)

        for shard_name in sorted(model_files):
            shard: dict[str, torch.Tensor] = {}
            for name, filename in weight_map.items():
                if filename != shard_name:
                    continue
                merged, layout = merge_tensor(
                    name,
                    [state[name] for state in states],
                    replicated_patterns,
                    row_patterns,
                )
                shard[name] = merged.contiguous()
                counts[layout] += 1
                if layout == "identical_full":
                    identical_full_names.add(name)
                parameters += merged.numel()
            save_file(shard, temporary / shard_name, metadata={"format": "pt"})
            print(f"saved {shard_name}: {len(shard)} tensors", flush=True)
            del shard
        if parameters != expected_parameters:
            raise RuntimeError(
                f"Expected {expected_parameters} parameters, merged {parameters}"
            )
        expected_full_names = EXPECTED_UNDECLARED_FULL_REPLICAS & set(weight_map)
        if identical_full_names != expected_full_names:
            raise RuntimeError(
                "undeclared full-replica set mismatch: "
                f"expected={sorted(expected_full_names)} "
                f"actual={sorted(identical_full_names)}"
            )
        expected_bytes = int(index.get("metadata", {}).get("total_size", -1))
        # For this BF16 model the template's total_size is an additional guard.
        if expected_bytes > 0 and expected_bytes != parameters * 2:
            raise RuntimeError(
                f"Template total_size {expected_bytes} does not match BF16 merge {parameters * 2}"
            )
        shutil.copy2(index_path, temporary / index_path.name)

        verified = 0
        for shard_name in sorted(model_files):
            with safe_open(temporary / shard_name, framework="pt", device="cpu") as handle:
                expected_names = sorted(
                    name for name, filename in weight_map.items() if filename == shard_name
                )
                if sorted(handle.keys()) != expected_names:
                    raise RuntimeError(f"{shard_name}: exported key mismatch")
                for name in expected_names:
                    tensor = handle.get_tensor(name)
                    merged, _ = merge_tensor(
                        name,
                        [state[name] for state in states],
                        replicated_patterns,
                        row_patterns,
                    )
                    if tensor.shape != merged.shape or tensor.dtype != merged.dtype:
                        raise RuntimeError(f"{name}: exported shape/dtype mismatch")
                    if not torch.equal(tensor, merged):
                        raise RuntimeError(f"{name}: exported value mismatch")
                    verified += 1
        if verified != len(weight_map):
            raise RuntimeError(f"Verified {verified} tensors, expected {len(weight_map)}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "tp_size": tp_size,
        "tensors": len(weight_map),
        "parameters": parameters,
        "layouts": counts,
    }


def main() -> None:
    args = parse_args()
    result = export_checkpoint(
        args.checkpoint_dir,
        args.template_hf,
        args.output_hf,
        args.tp_size,
        args.expected_parameters,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
