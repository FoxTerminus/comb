"""Gated Memory Unit — ported from ``microsoft/ArchScale/lit_gpt/gated_memory_unit.py``.

The GMU fuses current hidden states with a persistent ``memory`` tensor
produced by the boundary Mamba layer (layer 8).  The wrapper preserves the
memory channel so it can be passed unchanged to downstream layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMU(nn.Module):
    """Gated Memory Unit:  ``out_proj(SiLU(in_proj(x)) * memory)``."""

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        bias: bool = False,
        use_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.use_norm = use_norm

        self.in_proj = nn.Linear(d_model, d_mem, bias=bias)
        self.out_proj = nn.Linear(d_mem, d_model, bias=bias)
        if use_norm:
            self.norm = nn.RMSNorm(d_mem, eps=1e-5)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=5 ** 0.5)

    def forward(
        self, hidden_states: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        # Align sequence length (decode: memory may be longer)
        if memory.shape[1] != hidden_states.shape[1]:
            memory = memory[:, -hidden_states.shape[1] :, :]

        projected = F.silu(self.in_proj(hidden_states))
        gated = projected * memory
        if self.use_norm:
            gated = self.norm(gated)
        return self.out_proj(gated)


class GMUWrapper(nn.Module):
    """Wrapper that returns ``(gmu_output, memory)`` so the memory tensor
    flows through to the next layer unchanged."""

    def __init__(self, d_model: int, d_mem: int, bias: bool = False,
                 use_norm: bool = False):
        super().__init__()
        self.gmu = GMU(d_model, d_mem, bias=bias, use_norm=use_norm)

    def forward(
        self, hidden_states: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gmu(hidden_states, memory), memory
