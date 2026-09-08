#!/usr/bin/env python3
"""True-TP entry point layered over the audited TP=1 reproduction trainer.

This wrapper deliberately leaves ``train_llama_repro.py`` unchanged so the
active single-rank run and its hash-locked recovery supervisor keep their exact
behavior.  It intercepts only DeepSpeed initialization for this process:

* apply the Comb-specific AutoTP adapter;
* preserve the logical batch size across TP ranks;
* load universal checkpoints; and
* attach universal layout metadata to every new checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from deepspeed.checkpoint.constants import UNIVERSAL_CHECKPOINT_INFO

from training import train_llama_repro as base
from training.true_tp_comb_adapter import (
    apply_true_tp_comb_adapter,
    comb_universal_checkpoint_info,
    configure_deepspeed_for_true_tp,
    synchronize_true_tp_replicated_state,
)


def should_load_universal_checkpoint(
    resume_root_value: str | None, resume_tag_value: str | None
) -> bool:
    if resume_root_value is None:
        return False
    resume_root = Path(resume_root_value).resolve()
    if resume_tag_value is not None:
        return (
            resume_root
            / resume_tag_value
            / "comb_universal_conversion_complete.json"
        ).is_file()
    return (resume_root / "latest_universal").is_file()


def install_true_tp_hooks() -> None:
    original_initialize = base.deepspeed.initialize

    def initialize_true_tp(*args, **kwargs):
        if args:
            raise TypeError("true-TP wrapper requires keyword DeepSpeed initialization")
        model = kwargs["model"]
        original_config = kwargs["config"]
        tp_size = int(
            original_config["tensor_parallel"]["tensor_parallel"]["tp_size"]
        )
        if tp_size <= 1:
            raise ValueError(
                "train_llama_true_tp_repro.py requires --tensor-parallel-size > 1"
            )

        def cli_value(flag: str):
            for index, value in enumerate(sys.argv):
                if value == flag and index + 1 < len(sys.argv):
                    return sys.argv[index + 1]
                if value.startswith(flag + "="):
                    return value.split("=", 1)[1]
            return None

        output_value = cli_value("--output-dir")
        if output_value is None:
            raise ValueError("true TP requires --output-dir")
        output_dir = Path(output_value).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_root_value = cli_value("--resume-root")
        resume_tag_value = cli_value("--resume-tag")

        # tp_model_init performs an NCCL broadcast before DeepSpeedEngine is
        # constructed, so set the rank-local device explicitly first.  Without
        # this both launcher ranks use cuda:0 and NCCL reports a duplicate GPU.
        local_device = base.torch.distributed.get_rank() % base.torch.cuda.device_count()
        base.torch.cuda.set_device(local_device)
        configured = configure_deepspeed_for_true_tp(original_config, tp_size)
        load_universal = should_load_universal_checkpoint(
            resume_root_value, resume_tag_value
        )
        configured.setdefault("checkpoint", {})[
            "load_universal"
        ] = load_universal
        configured.setdefault("tensorboard", {})["output_path"] = str(output_dir)

        if base.torch.distributed.get_rank() == 0:
            source_paths = {
                "training/train_llama_true_tp_repro.py": Path(__file__).resolve(),
                "training/true_tp_comb_adapter.py": Path(__file__).with_name(
                    "true_tp_comb_adapter.py"
                ).resolve(),
                "training/train_llama_repro.py": Path(base.__file__).resolve(),
            }
            runtime_manifest = {
                "created_unix": time.time(),
                "pid": os.getpid(),
                "argv": sys.argv,
                "tp_size": tp_size,
                "logical_train_batch_size": original_config["train_batch_size"],
                "deepspeed_nominal_train_batch_size": configured[
                    "train_batch_size"
                ],
                "resume_root": resume_root_value,
                "resume_tag": resume_tag_value,
                "load_universal_checkpoint": load_universal,
                "source_sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in source_paths.items()
                },
            }
            manifest_path = output_dir / f"true_tp_runtime_manifest_{os.getpid()}.json"
            manifest_path.write_text(json.dumps(runtime_manifest, indent=2) + "\n")

            original_append_loss = base.append_loss
            timing_path = output_dir / "true_tp_step_timing.jsonl"

            def append_loss_with_timing(path, row):
                original_append_loss(path, row)
                with timing_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "wall_time": time.time(),
                                "optimizer_step": int(row[0]),
                            }
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())

            base.append_loss = append_loss_with_timing

        kwargs["model"] = apply_true_tp_comb_adapter(
            model, tp_size=tp_size, dtype=base.torch.bfloat16
        )
        kwargs["config"] = configured
        result = original_initialize(**kwargs)
        engine = result[0]
        if engine.dp_world_size != 1:
            raise ValueError(
                "this Comb true-TP entry point currently supports pure TP only "
                "(launcher world size must equal TP size)"
            )

        # Experimental post-step synchronization of ZeRO optimizer fragments is
        # deliberately opt-in.  A full-model diagnostic showed that treating
        # per-parameter ``_hp_mapping`` fragments as independent tensors can
        # corrupt the flattened ZeRO state.  Stage 0 needs no such repair: each
        # TP rank owns a complete optimizer for its local TP shard, while the
        # gradient hooks above keep logically replicated parameters identical.
        if os.environ.get("COMB_EXPERIMENTAL_ZERO_REPLICA_STATE_SYNC") == "1":
            original_load_checkpoint = engine.load_checkpoint

            def load_checkpoint_with_replica_sync(*load_args, **load_kwargs):
                loaded = original_load_checkpoint(*load_args, **load_kwargs)
                synchronize_true_tp_replicated_state(engine.module, tp_size)
                return loaded

            engine.load_checkpoint = load_checkpoint_with_replica_sync

            original_step = engine.step

            def step_with_replica_sync(*step_args, **step_kwargs):
                before = int(engine.global_steps)
                value = original_step(*step_args, **step_kwargs)
                if int(engine.global_steps) != before:
                    synchronize_true_tp_replicated_state(engine.module, tp_size)
                return value

            engine.step = step_with_replica_sync

        # The audited trainer passes CUDA local rank and launcher world size
        # into DatasetBase.bucketing().  Those are data-parallel coordinates
        # only in its TP=1 path.  Pure TP ranks must consume identical parquet
        # files, so all of them use the sole DP coordinate (rank=0, world=1).
        for dataset_class in set(base.DATASET_DICT.values()):
            original_bucketing = dataset_class.bucketing

            def bucketing_for_tp(
                dataset, _local_rank, _world_size, *, _original=original_bucketing
            ):
                return _original(dataset, 0, 1)

            dataset_class.bucketing = bucketing_for_tp

        original_save_checkpoint = engine.save_checkpoint

        def save_checkpoint_with_universal_info(
            save_dir,
            tag=None,
            client_state=None,
            save_latest=True,
            exclude_frozen_parameters=False,
        ):
            state = copy.deepcopy(client_state) if client_state is not None else {}
            expected = comb_universal_checkpoint_info()
            existing = state.get(UNIVERSAL_CHECKPOINT_INFO)
            if existing is not None and existing != expected:
                raise RuntimeError("conflicting Comb universal checkpoint metadata")
            state[UNIVERSAL_CHECKPOINT_INFO] = expected
            return original_save_checkpoint(
                save_dir,
                tag=tag,
                client_state=state,
                save_latest=save_latest,
                exclude_frozen_parameters=exclude_frozen_parameters,
            )

        engine.save_checkpoint = save_checkpoint_with_universal_info
        return result

    base.deepspeed.initialize = initialize_true_tp


def main() -> None:
    install_true_tp_hooks()
    base.main()


if __name__ == "__main__":
    main()
