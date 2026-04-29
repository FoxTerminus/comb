"""Factories for paper-aligned ArchScale SambaY and Samba+YOCO models.

The model internals are vendored from Microsoft ArchScale. This file keeps the
repo-facing surface small: choose an architecture alias, get the official d=8
configuration, and instantiate the official ``lit_gpt.model.GPT``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from baselines.ArchScale.vendor_path import ensure_archscale_on_path

ensure_archscale_on_path()

from lit_gpt.config import Config, get_parameters_count  # noqa: E402

if TYPE_CHECKING:
    from lit_gpt.model import GPT


ARCHITECTURE_TO_CONFIG = {
    "sambay": "sambay_d8",
    "sambay_d8": "sambay_d8",
    "samba_y": "sambay_d8",
    "sambayoco": "sambayoco_d8",
    "samba+yoco": "sambayoco_d8",
    "samba_yoco": "sambayoco_d8",
    "sambayoco_d8": "sambayoco_d8",
}


def canonical_config_name(architecture: str) -> str:
    key = architecture.lower()
    if key not in ARCHITECTURE_TO_CONFIG:
        known = ", ".join(sorted(ARCHITECTURE_TO_CONFIG))
        raise ValueError(f"Unknown ArchScale architecture {architecture!r}. Known aliases: {known}")
    return ARCHITECTURE_TO_CONFIG[key]


def build_config(
    architecture: str,
    *,
    block_size: int | None = None,
    vocab_size: int | None = None,
    padding_multiple: int | None = None,
    use_flash_attention: bool | None = None,
    **overrides: Any,
) -> Config:
    """Build a paper d=8 config with optional training-time overrides.

    By default this preserves the ArchScale paper setting: 32K vocabulary,
    untied embeddings, 128-token SWA window, Mamba self-decoder, and
    YOCO-style shared KV. Override ``vocab_size`` only when intentionally
    changing the tokenizer; doing so changes total parameters but keeps
    non-embedding architecture parameters paper-aligned.
    """

    config_name = canonical_config_name(architecture)
    kwargs: dict[str, Any] = dict(overrides)
    if block_size is not None:
        kwargs["block_size"] = block_size
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size
    if padding_multiple is not None:
        kwargs["padding_multiple"] = padding_multiple
    if use_flash_attention is not None:
        kwargs["fa2"] = use_flash_attention
    return Config.from_name(config_name, **kwargs)


def build_model(
    architecture: str,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    reset_parameters: bool = True,
    **config_overrides: Any,
) -> GPT:
    """Instantiate an ArchScale GPT model for SambaY or Samba+YOCO."""

    from lit_gpt.model import GPT

    config = build_config(architecture, **config_overrides)
    model = GPT(config)
    if reset_parameters:
        model.reset_parameters()
    if dtype is not None:
        model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device)
    return model


def describe_layer_schedule(config_or_architecture: Config | str) -> list[dict[str, Any]]:
    """Return the official YOCO/SambaY layer roles implied by ArchScale flags."""

    config = (
        build_config(config_or_architecture)
        if isinstance(config_or_architecture, str)
        else config_or_architecture
    )
    schedule = []
    for layer_idx in range(config.n_layer):
        use_rnn = False
        use_gmu = False
        gmu_save = False
        yoco_kv = False
        yoco_cross = False
        use_full = False

        if config.yoco:
            if layer_idx < config.n_layer // 2:
                use_rnn = config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0
            else:
                if config.gmu_yoco and not config.gmu_attn:
                    gmu_save = layer_idx >= (config.n_layer // 2)
                yoco_kv = layer_idx >= (config.n_layer // 2 + 1)
                yoco_cross = layer_idx >= (config.n_layer // 2 + 2)
                use_full = layer_idx >= (config.n_layer // 2 + 1)
                if layer_idx == (config.n_layer // 2):
                    use_rnn = config.rnn_per_layer > 0
                if config.gmu_yoco and layer_idx >= (config.n_layer // 2 + 2):
                    use_gmu = layer_idx % config.gmu_per_layer == 0
        else:
            use_rnn = config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0

        if use_gmu:
            mixer = "gmu"
        elif use_rnn:
            mixer = config.rnn_type
        elif yoco_cross:
            mixer = "cross_attention"
        elif use_full:
            mixer = "full_attention"
        else:
            mixer = "sliding_window_attention" if config.local_window > 0 else "attention"

        schedule.append(
            {
                "layer": layer_idx,
                "mixer": mixer,
                "use_rnn": use_rnn,
                "use_gmu": use_gmu,
                "gmu_save": gmu_save,
                "yoco_kv": yoco_kv,
                "yoco_cross": yoco_cross,
            }
        )
    return schedule


def config_summary(config: Config, train_config_name: str = "scaling_mup") -> dict[str, Any]:
    """Summarize the size-relevant fields of an ArchScale config."""

    return {
        "name": config.name,
        "n_layer": config.n_layer,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_query_groups": config.n_query_groups,
        "head_size": config.head_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "padded_vocab_size": config.padded_vocab_size,
        "tied_embed": config.tied_embed,
        "estimated_non_embedding_params": int(
            get_parameters_count(config.name, config.n_layer, config, train_config_name)
        ),
    }


def model_summary(model: GPT) -> dict[str, Any]:
    from lit_gpt.utils import num_parameters

    summary = config_summary(model.config)
    summary["actual_total_params"] = int(num_parameters(model))
    summary["config"] = asdict(model.config)
    return summary
