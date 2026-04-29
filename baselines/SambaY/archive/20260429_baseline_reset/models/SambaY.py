"""SambaY-Llama baseline.

This module implements the correctness-first SambaY path used by the local
baseline. It follows the paper-level data flow:

- Samba-style self-decoder layers produce hidden states and GMU memory.
- A single boundary full-attention projection produces shared K/V memory.
- The cross-decoder interleaves GMU blocks and shared-KV cross-attention.

The first implementation intentionally uses pure PyTorch attention and a small
Mamba-like token mixer so CPU tiny tests can validate architecture and data
contracts before CUDA kernels and TP are introduced.
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    repeat = None


_LLAMA_CONFIG_INIT_KEYS = {
    name
    for name in inspect.signature(LlamaConfig.__init__).parameters
    if name not in {"self", "kwargs"}
}


def _compute_loss(logits, labels, shift_labels, vocab_size):
    if labels is None and shift_labels is None:
        return None
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    if shift_labels is not None:
        if logits.shape[:-1] != shift_labels.shape:
            raise ValueError(
                "shift_labels must match logits sequence shape: "
                f"{tuple(shift_labels.shape)} != {tuple(logits.shape[:-1])}"
            )
        return loss_fct(logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_logits.numel() == 0:
        return logits.new_zeros(())
    return loss_fct(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))


def _slice_logits(lm_head, hidden_states, logits_to_keep):
    if isinstance(logits_to_keep, torch.Tensor):
        logits = lm_head(hidden_states[:, logits_to_keep, :])
    elif logits_to_keep > 0:
        logits = lm_head(hidden_states[:, -logits_to_keep:, :])
    else:
        logits = lm_head(hidden_states)
    return logits.float()


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    return states.repeat_interleave(repeats, dim=2)


def _segment_ids_from_cu(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    if cu_seqlens is None:
        return torch.zeros(total_len, dtype=torch.long)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long).cpu()
    return torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.long), lengths)


def _attention_mask_from_cu(
    q_len: int,
    k_len: int,
    cu_q: Optional[torch.Tensor],
    cu_k: Optional[torch.Tensor],
    causal: bool,
    window_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    q_segments = _segment_ids_from_cu(cu_q, q_len).to(device)
    k_segments = _segment_ids_from_cu(cu_k, k_len).to(device)
    mask = q_segments[:, None] == k_segments[None, :]
    if causal:
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        if q_len == k_len:
            mask = mask & (k_pos <= q_pos)
        else:
            offset = max(k_len - q_len, 0)
            mask = mask & (k_pos <= (q_pos + offset))
    if window_size is not None and window_size > 0:
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        if q_len != k_len:
            q_pos = q_pos + max(k_len - q_len, 0)
        mask = mask & ((q_pos - k_pos) < window_size)
    return mask


class SambaYConfig(PretrainedConfig):
    """Configuration for the SambaY-Llama baseline."""

    model_type = "sambay_llama"
    sub_configs = {"text_config": LlamaConfig}

    def __init__(
        self,
        text_config: Optional[Union[LlamaConfig, dict]] = None,
        num_self_decoder_layers: int = 16,
        num_cross_decoder_layers: int = 16,
        rnn_per_layer: int = 2,
        gmu_per_layer: int = 2,
        sliding_window: int = 1024,
        use_nope: bool = True,
        pad_token_id: Optional[int] = 128004,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        llama_kwargs = {}
        if text_config is None:
            for key in list(kwargs):
                if key in _LLAMA_CONFIG_INIT_KEYS:
                    llama_kwargs[key] = kwargs.pop(key)
            vocab_size = llama_kwargs.get("vocab_size", LlamaConfig().vocab_size)
            if "pad_token_id" not in llama_kwargs and pad_token_id is not None and 0 <= pad_token_id < vocab_size:
                llama_kwargs["pad_token_id"] = pad_token_id

        if text_config is None:
            self.text_config = LlamaConfig(**llama_kwargs)
        elif isinstance(text_config, dict):
            self.text_config = LlamaConfig(**text_config)
        elif isinstance(text_config, LlamaConfig):
            self.text_config = text_config
        else:
            raise TypeError("text_config must be None, dict, or LlamaConfig")

        if (
            text_config is None
            and "num_hidden_layers" in llama_kwargs
            and num_self_decoder_layers == 16
            and num_cross_decoder_layers == 16
            and llama_kwargs["num_hidden_layers"] != 32
        ):
            num_self_decoder_layers = llama_kwargs["num_hidden_layers"] // 2
            num_cross_decoder_layers = llama_kwargs["num_hidden_layers"] - num_self_decoder_layers

        self.num_self_decoder_layers = num_self_decoder_layers
        self.num_cross_decoder_layers = num_cross_decoder_layers
        self.num_hidden_layers = num_self_decoder_layers + num_cross_decoder_layers
        self.rnn_per_layer = rnn_per_layer
        self.gmu_per_layer = gmu_per_layer
        self.sliding_window = sliding_window
        self.use_nope = use_nope
        self.gmu_memory_size = 2 * self.text_config.hidden_size

        if self.num_hidden_layers != self.text_config.num_hidden_layers:
            raise ValueError(
                "SambaY layer split must match text_config.num_hidden_layers: "
                f"{self.num_hidden_layers} != {self.text_config.num_hidden_layers}"
            )

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@dataclass
class SambaYDynamicCache:
    """SambaY cache for prefill plus token-by-token decode."""

    num_self_decoder_layers: int = 0
    window_size: int = 0
    shared_keys: Optional[torch.Tensor] = None
    shared_values: Optional[torch.Tensor] = None
    gmu_memory: Optional[torch.Tensor] = None
    mamba_sums: Optional[list[Optional[torch.Tensor]]] = None
    mamba_counts: Optional[list[int]] = None
    self_keys: Optional[list[Optional[torch.Tensor]]] = None
    self_values: Optional[list[Optional[torch.Tensor]]] = None
    conv_states: Optional[list[Optional[torch.Tensor]]] = None
    boundary_keys: Optional[torch.Tensor] = None
    boundary_values: Optional[torch.Tensor] = None
    seen_tokens: int = 0

    def __post_init__(self):
        if self.num_self_decoder_layers > 0:
            self.mamba_sums = [None] * self.num_self_decoder_layers
            self.mamba_counts = [0] * self.num_self_decoder_layers
            self.self_keys = [None] * self.num_self_decoder_layers
            self.self_values = [None] * self.num_self_decoder_layers
            self.conv_states = [None] * self.num_self_decoder_layers

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def set_memory(self, keys: torch.Tensor, values: torch.Tensor, gmu_memory: torch.Tensor) -> None:
        self.shared_keys = keys
        self.shared_values = values
        self.gmu_memory = gmu_memory
        self.seen_tokens = keys.shape[1]

    def append_memory(self, keys: torch.Tensor, values: torch.Tensor, gmu_memory: torch.Tensor) -> None:
        if self.shared_keys is None:
            self.set_memory(keys, values, gmu_memory)
            return
        self.shared_keys = torch.cat([self.shared_keys, keys], dim=1)
        self.shared_values = torch.cat([self.shared_values, values], dim=1)
        self.gmu_memory = torch.cat([self.gmu_memory, gmu_memory], dim=1)
        self.seen_tokens = self.shared_keys.shape[1]

    def get_mamba_state(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], int]:
        return self.mamba_sums[layer_idx], self.mamba_counts[layer_idx]

    def update_mamba_state(self, layer_idx: int, running_sum: torch.Tensor, count: int) -> None:
        self.mamba_sums[layer_idx] = running_sum
        self.mamba_counts[layer_idx] = count

    def get_self_attention_state(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.self_keys[layer_idx], self.self_values[layer_idx]

    def update_self_attention_state(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if self.window_size > 0:
            keys = keys[:, -self.window_size :, :, :].contiguous()
            values = values[:, -self.window_size :, :, :].contiguous()
        self.self_keys[layer_idx] = keys
        self.self_values[layer_idx] = values

    def get_boundary_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.boundary_keys, self.boundary_values

    def update_boundary_state(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.boundary_keys = keys
        self.boundary_values = values


class SambaYPreTrainedModel(PreTrainedModel):
    """Base class for SambaY models."""

    config_class = SambaYConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["SambaYSelfDecoderLayer", "SambaYCrossDecoderLayer"]

    def _init_weights(self, module):
        text_config = self.config.get_text_config()
        std = getattr(self.config, "initializer_range", text_config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


class SambaYPureAttention(nn.Module):
    """Pure PyTorch attention for tiny correctness runs."""

    def __init__(self, config: LlamaConfig, is_cross_attention: bool = False, window_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_repeats = self.num_heads // self.num_key_value_heads
        self.is_cross_attention = is_cross_attention
        self.window_size = window_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        if not is_cross_attention:
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def project_kv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states = self.k_proj(hidden_states).view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_key_value_heads, self.head_dim
        )
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        if key_states is None or value_states is None:
            key_states, value_states = self.project_kv(hidden_states)

        k = _repeat_kv(key_states, self.kv_repeats)
        v = _repeat_kv(value_states, self.kv_repeats)
        k_len = k.shape[1]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        mask = _attention_mask_from_cu(
            q_len,
            k_len,
            cu_seqlens_q,
            cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
            causal=causal,
            window_size=self.window_size,
            device=hidden_states.device,
        )
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        output = torch.einsum("bhqk,bkhd->bqhd", probs, v).reshape(batch_size, q_len, -1)
        return self.o_proj(output)


class SambaYArchScaleMambaCore(nn.Module):
    """ArchScale Mamba-1 core with ``gmu_save=True`` semantics.

    This follows ``microsoft/ArchScale/lit_gpt/mamba_simple.py`` for the
    non-incremental path: the GMU memory is the selective-scan output before
    SiLU gating and before ``out_proj``. That memory has size
    ``expand * d_model``, which is ``2 * d_model`` for Mamba-1.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        if rearrange is None or repeat is None:
            raise ImportError("einops is required for the ArchScale Mamba path.")
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        self.causal_conv1d_fn = causal_conv1d_fn
        self.selective_scan_fn = selective_scan_fn
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.activation = "silu"

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.ones(self.d_inner, d_state, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        if self.conv1d.bias is not None:
            fan_in = self.conv1d.in_channels * self.conv1d.kernel_size[0]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv1d.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.x_proj.weight, a=math.sqrt(5))

        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {self.dt_init}")
        dt = torch.exp(
            torch.rand(self.d_inner, dtype=torch.float32)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.copy_(inv_dt.to(dtype=self.dt_proj.bias.dtype))
        self.dt_proj.bias._no_reinit = True

        a = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log.copy_(torch.log(a))
        nn.init.ones_(self.D)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, _ = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        x, z = xz.chunk(2, dim=1)
        z = z.transpose(-1, -2).contiguous()

        x = self.causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, b_state, c_state = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        b_state = rearrange(b_state, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        c_state = rearrange(c_state, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        a_state = -torch.exp(self.A_log.float())
        y = self.selective_scan_fn(
            x,
            dt,
            a_state,
            b_state,
            c_state,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        memory = rearrange(y, "b d l -> b l d")
        mixed = torch.nn.functional.silu(z) * memory
        return self.out_proj(mixed), memory


class SambaYMambaMixer(nn.Module):
    """Mamba token mixer interface that exposes the GMU memory channel.

    The production target is the official Mamba selective-scan implementation.
    In environments without ``mamba_ssm`` this falls back to a deterministic
    SSM-style recurrence so CPU tests can still validate SambaY data flow.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.d_mem = 2 * self.hidden_size
        self.uses_official_mamba = False
        self.official_mamba = None
        try:
            self.official_mamba = SambaYArchScaleMambaCore(self.hidden_size)
            self.uses_official_mamba = True
        except Exception:
            self.official_mamba = None
        self.in_proj = nn.Linear(self.hidden_size, 2 * self.d_mem, bias=False)
        self.decay_proj = nn.Linear(self.hidden_size, self.d_mem, bias=False)
        self.out_proj = nn.Linear(self.d_mem, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.official_mamba is not None and hidden_states.is_cuda:
            return self.official_mamba(hidden_states)

        gate, value = self.in_proj(hidden_states).chunk(2, dim=-1)
        decay = torch.sigmoid(self.decay_proj(hidden_states))
        states = []
        state = torch.zeros(value.shape[0], value.shape[-1], dtype=value.dtype, device=value.device)
        for step in range(value.shape[1]):
            state = decay[:, step, :] * state + value[:, step, :]
            states.append(state)
        memory = torch.stack(states, dim=1)
        mixed = torch.nn.functional.silu(gate) * memory
        return self.out_proj(mixed), memory

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        running_sum: Optional[torch.Tensor],
        count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.official_mamba is not None and hidden_states.is_cuda:
            if count > 0 or running_sum is not None:
                raise NotImplementedError(
                    "Official CUDA Mamba incremental state update is not implemented. "
                    "Run generation with use_cache=False for exact full-sequence recomputation."
                )
            # Prefill uses the ArchScale gmu_save path. Token-by-token official
            # state update is a later vLLM/export concern; generation disables
            # cache on CUDA official Mamba to avoid silent state drift.
            mixed, memory = self.forward(hidden_states)
            return mixed, memory, memory[:, -1:, :].contiguous(), count + hidden_states.shape[1]

        gate, value = self.in_proj(hidden_states).chunk(2, dim=-1)
        if running_sum is None:
            running_sum = torch.zeros(
                value.shape[0],
                1,
                value.shape[-1],
                dtype=value.dtype,
                device=value.device,
            )
        decay = torch.sigmoid(self.decay_proj(hidden_states))
        states = []
        state = running_sum.squeeze(1)
        for step in range(value.shape[1]):
            state = decay[:, step, :] * state + value[:, step, :]
            states.append(state)
        memory = torch.stack(states, dim=1)
        mixed = torch.nn.functional.silu(gate) * memory
        next_sum = memory[:, -1:, :].contiguous()
        next_count = count + value.shape[1]
        return self.out_proj(mixed), memory, next_sum, next_count


class SambaYGMU(nn.Module):
    """Gated Memory Unit: out_proj(SiLU(in_proj(x)) * memory)."""

    def __init__(self, d_model: int, d_mem: int, use_norm: bool = False, eps: float = 1e-5):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_mem, bias=False)
        self.out_proj = nn.Linear(d_mem, d_model, bias=False)
        self.norm = LlamaRMSNorm(d_mem, eps=eps) if use_norm else None

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if memory.shape[1] != hidden_states.shape[1]:
            memory = memory[:, -hidden_states.shape[1] :, :]
        projected = torch.nn.functional.silu(self.in_proj(hidden_states))
        if memory.shape[-1] != projected.shape[-1]:
            tp_group = getattr(self, "_tp_group", None)
            if tp_group is None or not dist.is_available() or not dist.is_initialized():
                raise ValueError(
                    "GMU memory width must match projected width unless tensor-parallel "
                    f"memory slicing is configured: memory={memory.shape[-1]}, projected={projected.shape[-1]}"
                )
            tp_size = dist.get_world_size(tp_group)
            tp_rank = dist.get_rank(tp_group)
            if memory.shape[-1] % tp_size != 0:
                raise ValueError(f"GMU memory width {memory.shape[-1]} is not divisible by tp_size={tp_size}")
            shard = memory.shape[-1] // tp_size
            if shard != projected.shape[-1]:
                raise ValueError(
                    f"GMU memory shard width {shard} does not match projected width {projected.shape[-1]}"
                )
            memory = memory[..., tp_rank * shard : (tp_rank + 1) * shard].contiguous()
        gated = projected * memory
        if self.norm is not None:
            gated = self.norm(gated)
        return self.out_proj(gated)


class SambaYGMUWrapper(nn.Module):
    """ArchScale-style GMU wrapper that preserves the memory object."""

    def __init__(self, d_model: int, d_mem: int, use_norm: bool = False, eps: float = 1e-5):
        super().__init__()
        self.gmu = SambaYGMU(d_model, d_mem, use_norm=use_norm, eps=eps)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gmu(hidden_states, memory), memory


class SambaYSelfDecoderLayer(nn.Module):
    """Self-decoder layer using either Mamba-like mixing or SWA."""

    def __init__(self, config: LlamaConfig, layer_idx: int, use_mamba: bool, window_size: int):
        super().__init__()
        self.use_mamba = use_mamba
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_mixer = SambaYMambaMixer(config) if use_mamba else SambaYPureAttention(config, window_size=window_size)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        layer_idx: Optional[int] = None,
        past_key_values: Optional[SambaYDynamicCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        gmu_memory = None
        if self.use_mamba:
            if use_cache and past_key_values is not None and layer_idx is not None:
                running_sum, count = past_key_values.get_mamba_state(layer_idx)
                mixed, gmu_memory, next_sum, next_count = self.token_mixer.forward_with_cache(
                    normed,
                    running_sum,
                    count,
                )
                past_key_values.update_mamba_state(layer_idx, next_sum, next_count)
            else:
                mixed, gmu_memory = self.token_mixer(normed)
        else:
            if use_cache and past_key_values is not None and layer_idx is not None:
                key_states, value_states = self.token_mixer.project_kv(normed)
                past_keys, past_values = past_key_values.get_self_attention_state(layer_idx)
                if past_keys is not None:
                    key_states = torch.cat([past_keys, key_states], dim=1)
                    value_states = torch.cat([past_values, value_states], dim=1)
                    cu_seqlens_k = torch.tensor(
                        [0, key_states.shape[1]],
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                else:
                    cu_seqlens_k = cu_seqlens_q
                mixed = self.token_mixer(
                    normed,
                    key_states=key_states,
                    value_states=value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    causal=True,
                )
                past_key_values.update_self_attention_state(layer_idx, key_states, value_states)
            else:
                mixed = self.token_mixer(normed, cu_seqlens_q=cu_seqlens_q, causal=True)
        hidden_states = residual + mixed

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, gmu_memory


class SambaYBoundaryFullAttentionLayer(nn.Module):
    """YOCO boundary full-attention layer that also produces shared K/V."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SambaYPureAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        past_key_values: Optional[SambaYDynamicCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        key_states, value_states = self.self_attn.project_kv(normed)

        if use_cache and past_key_values is not None:
            past_keys, past_values = past_key_values.get_boundary_state()
            if past_keys is not None:
                attn_keys = torch.cat([past_keys, key_states], dim=1)
                attn_values = torch.cat([past_values, value_states], dim=1)
                cu_seqlens_k = torch.tensor(
                    [0, attn_keys.shape[1]],
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            else:
                attn_keys = key_states
                attn_values = value_states
                cu_seqlens_k = cu_seqlens_q
            mixed = self.self_attn(
                normed,
                key_states=attn_keys,
                value_states=attn_values,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                causal=True,
            )
            past_key_values.update_boundary_state(attn_keys, attn_values)
        else:
            mixed = self.self_attn(normed, cu_seqlens_q=cu_seqlens_q, causal=True)

        hidden_states = residual + mixed
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, key_states, value_states


class SambaYSelfDecoder(nn.Module):
    """Samba-style self-decoder plus one shared KV producer."""

    def __init__(self, config: SambaYConfig):
        super().__init__()
        text_config = config.get_text_config()
        self.layers = nn.ModuleList(
            [
                SambaYSelfDecoderLayer(
                    text_config,
                    layer_idx,
                    use_mamba=(config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0),
                    window_size=config.sliding_window,
                )
                for layer_idx in range(config.num_self_decoder_layers)
            ]
        )
        self.kv_layer_norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.gmu_save_layer = SambaYSelfDecoderLayer(
            text_config,
            config.num_self_decoder_layers,
            use_mamba=True,
            window_size=config.sliding_window,
        )
        self.boundary_layer = SambaYBoundaryFullAttentionLayer(text_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        past_key_values: Optional[SambaYDynamicCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gmu_memory = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, maybe_memory = layer(
                hidden_states,
                cu_seqlens_q,
                layer_idx=layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if maybe_memory is not None:
                gmu_memory = maybe_memory
        hidden_states, gmu_memory = self.gmu_save_layer(
            hidden_states,
            cu_seqlens_q,
            layer_idx=len(self.layers),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if gmu_memory is None:
            raise RuntimeError("SambaY gmu_save layer did not produce GMU memory.")
        hidden_states, key_states, value_states = self.boundary_layer(
            hidden_states,
            cu_seqlens_q,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return hidden_states, key_states, value_states, gmu_memory


class SambaYCrossDecoderLayer(nn.Module):
    """Cross-decoder layer: either GMU or shared-KV cross attention."""

    def __init__(self, config: LlamaConfig, use_gmu: bool, d_mem: int):
        super().__init__()
        self.use_gmu = use_gmu
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_mixer = (
            SambaYGMU(config.hidden_size, d_mem)
            if use_gmu
            else SambaYPureAttention(config, is_cross_attention=True)
        )
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        gmu_memory: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if self.use_gmu:
            mixed = self.token_mixer(normed, gmu_memory)
        else:
            mixed = self.token_mixer(
                normed,
                key_states=key_states,
                value_states=value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                causal=True,
            )
        hidden_states = residual + mixed

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class SambaYCrossDecoder(nn.Module):
    """Cross-decoder with GMU/cross-attention interleaving."""

    def __init__(self, config: SambaYConfig):
        super().__init__()
        text_config = config.get_text_config()
        self.layers = nn.ModuleList()
        layer_count = max(config.num_cross_decoder_layers - 2, 0)
        for layer_idx in range(layer_count):
            global_idx = config.num_self_decoder_layers + 2 + layer_idx
            use_gmu = config.gmu_per_layer > 0 and global_idx % config.gmu_per_layer == 0
            self.layers.append(SambaYCrossDecoderLayer(text_config, use_gmu, config.gmu_memory_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        gmu_memory: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                key_states,
                value_states,
                gmu_memory,
                cu_seqlens_q,
                cu_seqlens_k,
            )
        return hidden_states


class SambaYTextModel(SambaYPreTrainedModel):
    """SambaY text model."""

    config_class = SambaYConfig
    base_model_prefix = "model"

    def __init__(self, config: SambaYConfig):
        text_config = config.get_text_config()
        super().__init__(config)
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.self_decoder = SambaYSelfDecoder(config)
        self.cross_decoder = SambaYCrossDecoder(config)
        self.norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        past_key_values: Optional[SambaYDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, inputs_embeds.shape[1]], dtype=torch.int32, device=inputs_embeds.device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if cu_seqlens_q.numel() != 2:
                raise NotImplementedError("SambaY cache mode currently supports one packed sequence.")
            if past_key_values is None:
                past_key_values = SambaYDynamicCache(
                    num_self_decoder_layers=self.config.num_self_decoder_layers + 1,
                    window_size=self.config.sliding_window,
                )
            elif not isinstance(past_key_values, SambaYDynamicCache):
                if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
                    past_key_values = SambaYDynamicCache(
                        num_self_decoder_layers=self.config.num_self_decoder_layers + 1,
                        window_size=self.config.sliding_window,
                    )
                else:
                    raise TypeError("past_key_values must be a SambaYDynamicCache for SambaY cache mode.")
            previous_memory_len = past_key_values.get_seq_length()
        else:
            previous_memory_len = 0

        hidden_states, key_states, value_states, gmu_memory = self.self_decoder(
            inputs_embeds,
            cu_seqlens_q,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            if previous_memory_len == 0:
                past_key_values.set_memory(key_states, value_states, gmu_memory)
            else:
                past_key_values.append_memory(key_states, value_states, gmu_memory)
            key_states = past_key_values.shared_keys
            value_states = past_key_values.shared_values
            gmu_memory = past_key_values.gmu_memory

        cu_seqlens_k = cu_seqlens_q
        if key_states.shape[1] != hidden_states.shape[1]:
            cu_seqlens_k = torch.tensor([0, key_states.shape[1]], dtype=torch.int32, device=hidden_states.device)

        hidden_states = self.cross_decoder(
            hidden_states,
            key_states,
            value_states,
            gmu_memory,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class SambaYForCausalLM(SambaYPreTrainedModel, GenerationMixin):
    """Causal LM wrapper for SambaY."""

    config_class = SambaYConfig
    _supports_static_cache = False
    base_model_prefix = "language_model"

    def __init__(self, config: SambaYConfig):
        text_config = config.get_text_config()
        super().__init__(config)
        self.text_config = text_config
        self.vocab_size = text_config.vocab_size
        self.model = SambaYTextModel._from_config(config)
        self.lm_head = nn.Linear(text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _uses_cuda_official_mamba(self, device: torch.device) -> bool:
        if device.type != "cuda":
            return False
        layers = list(self.model.self_decoder.layers) + [self.model.self_decoder.gmu_save_layer]
        return any(
            layer.use_mamba and getattr(layer.token_mixer, "official_mamba", None) is not None
            for layer in layers
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("SambaY generation requires input_ids or inputs_embeds.")
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if batch_size != 1:
            raise NotImplementedError("SambaY generation currently supports batch_size=1.")

        if past_key_values is not None and not isinstance(past_key_values, SambaYDynamicCache):
            if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
                past_key_values = None
            else:
                raise TypeError("SambaY generation requires SambaYDynamicCache when past_key_values is provided.")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        requested_cache = kwargs.get("use_cache", True)
        use_cache = requested_cache
        if requested_cache and self._uses_cuda_official_mamba(device):
            use_cache = False
            past_key_values = None

        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            else:
                inputs_embeds = inputs_embeds[:, -1:, :]
            seq_len = 1
        else:
            seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        model_inputs = {
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cu_seqlens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            "max_seqlen_q": seq_len,
        }
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if beam_idx.numel() != 1 or beam_idx.item() != 0:
            raise NotImplementedError("SambaY cache reordering is only implemented for batch_size=1.")
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, SambaYDynamicCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        shift_labels: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        logits = _slice_logits(self.lm_head, outputs.last_hidden_state, logits_to_keep)
        loss_logits = logits
        if labels is not None and shift_labels is None and not (
            isinstance(logits_to_keep, int) and logits_to_keep == 0
        ):
            loss_logits = _slice_logits(self.lm_head, outputs.last_hidden_state, 0)
        loss = _compute_loss(loss_logits, labels, shift_labels, self.vocab_size)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
