"""GPT model with YOCO decoder-decoder architecture and SambaY hybrid layers.

Ported from ``microsoft/ArchScale/lit_gpt/model.py``.  The 16-layer model
is split into two halves:

Self-decoder (layers 0-7)
    Alternating Mamba (even layers) and local sliding-window attention
    (odd layers).  Produces hidden states for the cross-decoder.

Cross-decoder (layers 8-15)
    - Layer 8  → boundary Mamba with ``gmu_save=True`` (produces GMU memory)
    - Layer 9  → full attention (produces shared KV cache)
    - Layers 10,12,14 → GMU (if ``gmu_yoco=True``, else cross-attn)
    - Layers 11,13,15 → cross-attention (reads shared KV cache)
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import Config
from .gmu import GMUWrapper
from .mamba import Mamba


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network::

        out = w3(SiLU(w1(x)) * w2(x))
    """

    def __init__(self, n_embd: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.w2 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.w3 = nn.Linear(intermediate_size, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# RMS Normalisation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """A single transformer-like block whose token-mixer type is determined
    by its *layer_idx* and the YOCO / SambaY configuration."""

    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        cfg = copy.deepcopy(config)
        n_embd = cfg.n_embd
        mid = cfg.n_layer // 2  # boundary between self- and cross-decoder

        # --- YOCO role detection ------------------------------------------
        self.use_rnn: bool = False
        self.use_gmu: bool = False
        self.yoco_kv: bool = False
        self.yoco_cross: bool = False
        self.use_full: bool = False

        if not cfg.yoco:
            raise NotImplementedError(
                "Non-YOCO models are not supported in this implementation."
            )

        if layer_idx < mid:
            # ============== SELF-DECODER (layers 0 .. mid-1) ==============
            self.use_rnn = (
                cfg.rnn_per_layer > 0
                and layer_idx % cfg.rnn_per_layer == 0
            )
        else:
            # ============== CROSS-DECODER (layers mid .. n_layer-1) ========
            self.yoco_kv = layer_idx >= (mid + 1)     # layers >= 9
            self.yoco_cross = layer_idx >= (mid + 2)   # layers >= 10
            self.use_full = layer_idx >= (mid + 1)      # layers >= 9

            if layer_idx == mid:
                self.use_rnn = cfg.rnn_per_layer > 0

            if (
                cfg.gmu_yoco
                and not cfg.gmu_attn
                and not cfg.gmu_mlp
                and layer_idx >= (mid + 2)
                and cfg.rnn_per_layer > 0
                and cfg.rnn_type in ("mamba", "mamba2", "gdn")
            ):
                self.use_gmu = layer_idx % cfg.gmu_per_layer == 0

        # --- normalisation --------------------------------------------------
        self.norm_1 = RMSNorm(n_embd, eps=cfg.norm_eps)
        self.norm_2 = RMSNorm(n_embd, eps=cfg.norm_eps)

        # --- token mixer ----------------------------------------------------
        if self.use_gmu:
            gmu_inner = n_embd * 2
            self.token_mixer = GMUWrapper(n_embd, gmu_inner, bias=cfg.bias)
        elif self.use_rnn:
            gmu_save_flag = (layer_idx == mid)  # boundary Mamba saves GMU
            self.token_mixer = Mamba(
                n_embd,
                gmu_save=gmu_save_flag,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        else:
            self.token_mixer = CausalSelfAttention(
                n_embd=n_embd,
                n_head=cfg.n_head,
                n_query_groups=cfg.n_query_groups,
                head_dim=cfg.head_dim,
                yoco_cross=self.yoco_cross,
                local_window=cfg.local_window if not self.use_full else -1,
                bias=cfg.attn_bias,
            )

        # --- MLP ----------------------------------------------------------
        self.mlp = SwiGLUMLP(n_embd, cfg.intermediate_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gmu_mems: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # ---- token mixing -------------------------------------------------
        residual = x
        normed = self.norm_1(x)

        if self.use_rnn:
            # Mamba: returns (output, gmu_mems_or_None)
            h, new_gmu = self.token_mixer(normed)
            new_kv_cache = kv_cache
        elif self.use_gmu:
            # GMU: returns (output, memory)
            h, new_gmu = self.token_mixer(normed, gmu_mems)
            new_kv_cache = kv_cache
        else:
            # Attention: returns (output, (k, v))
            h, new_kv = self.token_mixer(normed, kv_cache=kv_cache)
            new_kv_cache = new_kv
            new_gmu = gmu_mems

        x = residual + h.to(x.dtype)

        # ---- MLP ----------------------------------------------------------
        residual = x
        x = residual + self.mlp(self.norm_2(x)).to(x.dtype)

        return x, new_kv_cache, new_gmu


# ---------------------------------------------------------------------------
# GPT (full model)
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """SambaY / Samba+YOCO language model.

    Parameters
    ----------
    config:
        Model configuration (``Config`` dataclass).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(
            config.padded_vocab_size, config.n_embd
        )
        self.blocks = nn.ModuleList(
            Block(config, i) for i in range(config.n_layer)
        )
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=False
        )

        # Tie weights if requested
        if config.tied_embed:
            self.lm_head.weight = self.wte.weight

    def reset_parameters(self) -> None:
        """Initialize weights (called once after meta-device init)."""
        std = 0.02
        nn.init.normal_(self.wte.weight, mean=0.0, std=std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            Long tensor of shape ``(B, T)``.
        labels:
            Long tensor of shape ``(B, T)`` for next-token prediction.
            Standard causal shift is applied internally.

        Returns
        -------
        dict with keys ``logits`` and ``loss`` (if labels provided).
        """
        B, T = input_ids.shape
        x = self.wte(input_ids)

        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        gmu_mems: Optional[torch.Tensor] = None

        for block in self.blocks:
            x, kv_cache, gmu_mems = block(
                x, kv_cache=kv_cache, gmu_mems=gmu_mems
            )

        x = self.norm(x)
        logits = self.lm_head(x).float()

        result: dict = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.padded_vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def estimated_params(self) -> int:
        """Rough non-embedding parameter count."""
        return self.config.estimated_params()

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
