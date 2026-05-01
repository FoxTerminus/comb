"""Config dataclass for SambaY / Samba+YOCO model architectures.

Ported from `microsoft/ArchScale <https://github.com/microsoft/ArchScale>`_
``lit_gpt/config.py``.  Only the fields relevant to the sambay / sambayoco
config families are retained; MoE, skip-connection, and scaling fields are
stripped to keep the surface area small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def find_multiple(n: int, k: int) -> int:
    """Round *n* up to the nearest multiple of *k*."""
    return int(math.ceil(n / k)) * k


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Model configuration for YOCO / SambaY hybrid architectures."""

    # -- metadata -----------------------------------------------------------
    org: str = "Microsoft"
    name: str = "sambay_d16"

    # -- sequence / vocabulary ----------------------------------------------
    block_size: int = 65536         # training context length (64K)
    vocab_size: int = 32000         # Llama-2 tokenizer vocabulary
    padding_multiple: int = 64      # padded_vocab_size alignment
    padded_vocab_size: Optional[int] = None

    # -- layout -------------------------------------------------------------
    n_layer: int = 16               # total layers (self-decoder + cross-decoder)
    n_head: int = 16                # attention heads
    n_embd: int = 1984              # hidden dimension (ar * n_layer)
    ar: Optional[int] = 124         # aspect ratio → n_embd = ar * n_layer
    mlp_expand: int = 4             # intermediate_size = mlp_expand * n_embd
    intermediate_size: Optional[int] = None
    head_dim: int = 128             # per-head dimension
    n_query_groups: Optional[int] = 4  # GQA key-value groups (None → n_head)

    # -- embeddings / output ------------------------------------------------
    tied_embed: bool = False
    scale_embed: bool = False
    eos_token_id: int = 2

    # -- attention ----------------------------------------------------------
    rotary_percentage: float = 1.0
    rope_base: int = 10000
    ada_rope: bool = False
    separate_qkv: bool = True
    fa2: bool = True
    local_window: int = 128         # sliding-window size in self-decoder
    use_cu_seqlen: bool = False
    qk_norm: bool = False
    attn_norm: bool = False
    nope: bool = True               # no positional encoding (for Mamba layers)
    yoco_nope: bool = False
    use_da: bool = False            # differential attention
    sink_attn: bool = False
    gated_attn: bool = False
    relu2: bool = False
    full_swa_extend: bool = False
    scaling_factor: float = 1.0

    # -- bias ---------------------------------------------------------------
    bias: bool = False
    attn_bias: bool = False
    attn_out_bias: bool = False
    no_mlp_bias: bool = False

    # -- mlp ---------------------------------------------------------------
    mlp: bool = True
    oss_swiglu: bool = False
    ffn_norm: bool = False
    mlp_relu2: bool = False

    # -- hybrid (YOCO / SambaY) ---------------------------------------------
    full_per_layer: int = 1000000
    rnn_per_layer: int = 2          # insert Mamba every N layers in self-decoder
    rnn_type: str = "mamba"         # "mamba" | "mamba2" | "gdn" | "retnet" | "gla" | "delta"
    attn_layer_pos: Optional[str] = None
    yoco: bool = True               # enable YOCO decoder-decoder architecture
    gmu_yoco: bool = True           # enable Gated Memory Unit (False → Samba+YOCO)
    gmu_per_layer: int = 2          # insert GMU every N layers in cross-decoder
    gmu_attn: bool = False
    gmu_mlp: bool = False
    jamba_norm: bool = False
    yoco_window: bool = False

    # -- skip connections ---------------------------------------------------
    post_norm: bool = False
    decouple_postnorm: bool = False
    skip_gain: bool = False
    skip_weight_per_layer: int = -1
    residual_in_fp32: bool = True
    no_skip: bool = False
    sum_skip: bool = False

    # -- norms --------------------------------------------------------------
    _norm_class: str = "RMSNorm"
    norm_eps: float = 1e-5

    # -- initialisation -----------------------------------------------------
    w_init_scale: float = 1.0

    # ------------------------------------------------------------------
    # Post-initialisation (ported from ArchScale Config.__post_init__)
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )

        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0, (
                f"n_head ({self.n_head}) must be divisible by "
                f"n_query_groups ({self.n_query_groups})"
            )
        else:
            self.n_query_groups = self.n_head

        if self.intermediate_size is None:
            self.intermediate_size = self.mlp_expand * self.n_embd

        if self.ar is not None:
            self.n_embd = self.ar * self.n_layer
            self.intermediate_size = self.mlp_expand * self.n_embd

        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def head_size(self) -> int:
        """Effective per-head dimension (alias for head_dim)."""
        return self.head_dim

    @property
    def num_self_decoder_layers(self) -> int:
        """Number of layers in the self-decoder (bottom half)."""
        return self.n_layer // 2

    @property
    def num_cross_decoder_layers(self) -> int:
        """Number of layers in the cross-decoder (top half)."""
        return self.n_layer - self.n_layer // 2

    def estimated_params(self) -> int:
        """Rough parameter count using the μP++ formula from ArchScale.

        n ≈ n_mult × depth³   where  n_mult = 14.5·ar² + 144·ar (for sambay).
        """
        ar = self.ar if self.ar is not None else self.n_embd // self.n_layer
        n_mult = 14.5 * (ar ** 2) + 144 * ar
        return int(n_mult * (self.n_layer ** 3))

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Config":
        """Look up a preset config by name and create a Config instance.

        Supported names: ``sambay_d16``, ``sambayoco_d16``.
        """
        presets = _build_presets()
        if name not in presets:
            raise KeyError(
                f"Unknown config name {name!r}. "
                f"Known: {sorted(presets.keys())}"
            )
        cfg_dict = {**presets[name], **kwargs}
        return cls(**cfg_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Preset registry (sambay_d16, sambayoco_d16)
# ---------------------------------------------------------------------------

def _build_presets():
    """Return a dict of preset configs matching ArchScale's lit_gpt/config.py."""

    def _sambay(d: int) -> dict:
        return dict(
            org="Microsoft",
            name=f"sambay_d{d}",
            block_size=65536,
            vocab_size=32000,
            padding_multiple=64,
            rnn_per_layer=2,
            rnn_type="mamba",
            yoco=True,
            gmu_yoco=True,
            nope=True,
            n_layer=d,
            n_head=d,
            head_dim=128,
            ar=124,
            n_query_groups=d // 4,
            mlp_expand=4,
            local_window=128,
        )

    def _sambayoco(d: int) -> dict:
        return dict(
            org="Microsoft",
            name=f"sambayoco_d{d}",
            block_size=65536,
            vocab_size=32000,
            padding_multiple=64,
            rnn_per_layer=2,
            rnn_type="mamba",
            yoco=True,
            gmu_yoco=False,
            nope=True,
            n_layer=d,
            n_head=d,
            head_dim=128,
            ar=126,
            n_query_groups=d // 4,
            mlp_expand=4,
            local_window=128,
        )

    presets = {}
    for d in [8, 12, 16, 20, 24]:
        presets[f"sambay_d{d}"] = _sambay(d)
        presets[f"sambayoco_d{d}"] = _sambayoco(d)
    return presets
