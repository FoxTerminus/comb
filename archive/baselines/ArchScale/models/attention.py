"""Self-attention and cross-attention with sliding-window and GQA support.

Ported from ``microsoft/ArchScale/lit_gpt/attention.py``.  When
``yoco_cross=True`` the layer has no ``k_proj`` / ``v_proj`` — it reads
K,V from the shared ``kv_cache`` produced by the boundary full-attention
layer.

All attention paths enforce causal masking.  Local sliding-window layers
use FlashAttention's native ``window_size`` parameter to avoid
materialising a dense ``(T, T)`` mask at 64K context.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _fa2_func
except ImportError:
    _fa2_func = None


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat key-value heads for GQA."""
    if repeats == 1:
        return states
    return states.repeat_interleave(repeats, dim=2)


class CausalSelfAttention(nn.Module):
    """Multi-head (self- or cross-) attention with optional sliding window.

    All paths are **causal** — position ``t`` never attends to ``> t``.
    Local self-attention uses FlashAttention-2 ``window_size`` so it
    scales to 64K without materialising a full ``(T, T)`` mask.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_query_groups: int,
        head_dim: int,
        yoco_cross: bool = False,
        local_window: int = -1,
        bias: bool = False,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_dim = head_dim
        self.yoco_cross = yoco_cross
        self.local_window = local_window
        self.kv_repeats = n_head // n_query_groups
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=bias)
        if not yoco_cross:
            self.k_proj = nn.Linear(n_embd, n_query_groups * head_dim, bias=False)
            self.v_proj = nn.Linear(n_embd, n_query_groups * head_dim, bias=bias)
        self.proj = nn.Linear(n_head * head_dim, n_embd, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        # ---- Q ------------------------------------------------------------
        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim)

        # ---- K, V ---------------------------------------------------------
        if self.yoco_cross:
            assert kv_cache is not None, (
                "yoco_cross attention requires kv_cache"
            )
            k, v = kv_cache
        else:
            k = self.k_proj(x).view(B, T, self.n_query_groups, self.head_dim)
            v = self.v_proj(x).view(B, T, self.n_query_groups, self.head_dim)

        # save raw K,V for downstream cross-attention (before GQA repeat)
        raw_kv_cache = (k, v) if not self.yoco_cross else kv_cache

        # ---- GQA repeat ---------------------------------------------------
        k = _repeat_kv(k, self.kv_repeats)
        v = _repeat_kv(v, self.kv_repeats)

        # ---- attention ----------------------------------------------------
        if self.local_window > 0 and not self.yoco_cross:
            # local SWA: FlashAttention-2 window_size.
            # flash_attn_func expects (B, T, H, D) — NO transpose.
            if _fa2_func is None:
                raise RuntimeError(
                    "flash_attn required for local sliding-window attention"
                )
            # q, k, v are already (B, T, H, D) after view and GQA repeat
            y = _fa2_func(
                q, k, v,
                causal=True,
                window_size=(self.local_window - 1, 0),
                softmax_scale=self.scale,
            )
            y = y.reshape(B, T, -1)
        else:
            # full causal attention (self-attn or cross-attn).
            # SDPA expects (B, H, T, D) — need transpose.
            q_t = q.transpose(1, 2).contiguous()  # (B, H, T, D)
            k_t = k.transpose(1, 2).contiguous()
            v_t = v.transpose(1, 2).contiguous()
            y = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                dropout_p=0.0, is_causal=True, scale=self.scale,
            )
            y = y.transpose(1, 2).reshape(B, T, -1)
        y = self.proj(y)

        return y, raw_kv_cache
