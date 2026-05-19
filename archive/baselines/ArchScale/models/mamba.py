"""Mamba-1 selective-scan token mixer — ported from
``microsoft/ArchScale/lit_gpt/mamba_simple.py``.

When ``gmu_save=True`` the raw SSM output (before SwiGLU gating and
``out_proj``) is saved into ``gmu_mems`` so the cross-decoder GMU layers
can use it as gating memory.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class Mamba(nn.Module):
    """Mamba-1 layer (S6 / selective-scan).

    Parameters
    ----------
    d_model:
        Input / output feature dimension.
    d_state:
        SSM state dimension (default 16).
    d_conv:
        Convolution kernel width (default 4).
    expand:
        Inner-dimension multiplier (default 2 → d_inner = 2 * d_model).
    gmu_save:
        When True, store the pre-gate SSM output in ``gmu_mems`` so
        downstream GMU layers can consume it.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        gmu_save: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.gmu_save = gmu_save

        # projections
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, N)
        A = A.expand(self.d_inner, -1).contiguous()                         # (D, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

        self.reset_parameters()

        # init dt_proj so that softplus(dt_bias) ∈ [dt_min, dt_max]
        dt_init_std = self.dt_rank ** (-0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            dt = (
                torch.rand(self.d_inner, dtype=torch.float32)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # tag to skip re-init

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        if self.conv1d.bias is not None:
            fan = self.conv1d.in_channels * self.conv1d.kernel_size[0]
            nn.init.uniform_(self.conv1d.bias, -1.0 / math.sqrt(fan),
                             1.0 / math.sqrt(fan))
        nn.init.kaiming_uniform_(self.x_proj.weight, a=math.sqrt(5))
        nn.init.ones_(self.D)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        w = self.in_proj.weight.to(dtype=hidden_states.dtype)

        # ---- in-projection ------------------------------------------------
        xz = rearrange(
            w @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=L,
        )
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.unsqueeze(0).unsqueeze(-1)

        x, z = xz.chunk(2, dim=1)  # (B, d_inner, L) each
        z = rearrange(z, "b d l -> b l d")

        # ---- causal conv1d ------------------------------------------------
        x_conv = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation="silu",
        )
        x_conv = rearrange(x_conv, "b d l -> (b l) d")

        # ---- S6 parameters ------------------------------------------------
        x_dbl = self.x_proj(x_conv)  # (B*L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        B_ssm = rearrange(B_ssm, "(b l) n -> b n l", l=L).contiguous()
        C_ssm = rearrange(C_ssm, "(b l) n -> b n l", l=L).contiguous()

        # ---- selective scan -----------------------------------------------
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        y = selective_scan_fn(
            x_conv.view(B, L, -1).transpose(1, 2),  # (B, d_inner, L)
            dt,
            A,
            B_ssm,
            C_ssm,
            self.D.float(),
            z=None if self.gmu_save else rearrange(z, "b l d -> b d l"),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # y is (B, d_inner, L) → (B, L, d_inner)
        y = rearrange(y, "b d l -> b l d")

        # ---- gmu_save -----------------------------------------------------
        gmu_mems: Optional[torch.Tensor] = None
        if self.gmu_save:
            gmu_mems = y           # raw SSM output, pre-gate
            y = F.silu(rearrange(z, "b l d -> b l d")) * y
        else:
            # gating already applied inside selective_scan_fn via z
            pass

        # ---- out projection -----------------------------------------------
        out = self.out_proj(y)
        return out, gmu_mems
