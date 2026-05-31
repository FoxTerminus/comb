"""
Packed FlashAttention CombLlama with a single-sample benchmark decode path.

Training and validation loss use full-sequence packed prefill with
``flash_attn_varlen_func``. The chunk model is a causal Llama-style stack
initialized from the base decoder and only exports cross-attention K/V states
at the configured cross layers. Benchmark generation is intentionally narrower:
batch size 1, greedy decoding, prefill with packed FlashAttention, and
text self-attention decode with ``flash_attn_with_kvcache``.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn
from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.utils import auto_docstring, can_return_tuple, logging


logger = logging.get_logger(__name__)


def _ensure_packed_2d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == 2 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(f"`{name}` must have shape [total_tokens] or [1, total_tokens], got {tuple(tensor.shape)}")


def _require(name: str, value):
    if value is None:
        raise ValueError(f"`{name}` is required for packed FlashAttention.")
    return value


def _as_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _packed_loss(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    labels = labels.reshape(-1).to(logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, ignore_index=-100)


@dataclass
class CombLlamaDecodeCache:
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
    cache_seqlens: torch.Tensor
    max_cache_len: int


class CombLlamaConfig(PretrainedConfig):
    model_type = "combllama"
    attribute_map = {"chunk_token_id": "chunk_token_index"}
    sub_configs = {"text_config": LlamaConfig}

    def __init__(
        self,
        text_config: Optional[LlamaConfig] = None,
        chunk_token_index: int = 128255,
        num_hidden_layers: int = 40,
        cross_attention_layers: Optional[List[int]] = None,
        pad_token_id: Optional[int] = 128004,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 7, 11, 15, 19, 23, 27, 31]

        self.chunk_token_index = chunk_token_index
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers

        if text_config is None:
            self.text_config = LlamaConfig()
        elif isinstance(text_config, dict):
            self.text_config = LlamaConfig(**text_config)
        elif isinstance(text_config, LlamaConfig):
            self.text_config = text_config
        else:
            raise TypeError("`text_config` must be None, a dict, or a LlamaConfig.")

        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class CrossAttention(nn.Module):
    """Packed text-to-context attention: text Q attends chunk K/V."""

    is_causal = False

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        k, v = cross_attention_states
        k = self.k_norm(k)

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            softmax_scale=self.scaling,
            causal=False,
        )
        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        return self.o_proj(attn_output)


class ChunkFlashAttention(nn.Module):
    """Causal packed self-attention for context/chunk tokens."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q, k = q.squeeze(0), k.squeeze(0)

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            softmax_scale=self.scaling,
            causal=True,
        )
        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_dim))


class ChunkFlashLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ChunkFlashAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CombLlamaCrossAttentionDecoderLayer(nn.Module):
    """Cross-attention block with zero-initialized tanh gates."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = CrossAttention(config, layer_idx=layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))
        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states,
            cross_attention_states,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.cross_attn_mlp_gate.tanh() * hidden_states


@auto_docstring
class CombLlamaPreTrainedModel(PreTrainedModel):
    config_class = CombLlamaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["CombLlamaChunkModel", "CombLlamaCrossAttentionDecoderLayer", "LlamaDecoderLayer"]
    _supports_flash_attn = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, CombLlamaCrossAttentionDecoderLayer):
            module.cross_attn_attn_gate.data.zero_()
            module.cross_attn_mlp_gate.data.zero_()


class CombLlamaChunkModel(CombLlamaPreTrainedModel):
    """Causal Llama-style context stack that exports K/V states at cross layers."""

    config_class = CombLlamaConfig
    base_model_prefix = "chunk_model"

    def __init__(self, config: CombLlamaConfig):
        super().__init__(config)
        text_config = config.get_text_config()
        self.cross_attention_layers = config.cross_attention_layers
        self.num_cross_layers = len(config.cross_attention_layers)
        self.num_chunk_layers = config.num_hidden_layers - self.num_cross_layers
        self._cross_layer_map = {
            layer_idx: cross_id for cross_id, layer_idx in enumerate(self.cross_attention_layers)
        }
        self.hidden_size = text_config.hidden_size
        self.head_dim = self.hidden_size // text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.embed_tokens = nn.Embedding(text_config.vocab_size, self.hidden_size, text_config.pad_token_id)
        self.layers = nn.ModuleList([ChunkFlashLayer(text_config) for _ in range(self.num_chunk_layers)])
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.k_proj = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
                for _ in range(self.num_cross_layers)
            ]
        )
        self.v_proj = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
                for _ in range(self.num_cross_layers)
            ]
        )
        self.post_init()

    def forward(
        self,
        chunk_ids: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        position_ids_k: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        chunk_ids = _ensure_packed_2d("chunk_ids", chunk_ids)
        position_ids_k = _ensure_packed_2d("position_ids_k", position_ids_k)
        max_seqlen_k = _as_int(max_seqlen_k)
        hidden_states = self.embed_tokens(chunk_ids.long()).squeeze(0)
        position_embeddings = self.rotary_emb(hidden_states.unsqueeze(0), position_ids_k)

        cross_attention_states = [None] * self.num_cross_layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings, cu_seqlens_k, max_seqlen_k)
            cross_id = self._cross_layer_map.get(layer_idx)
            if cross_id is not None:
                key_states = self.k_proj[cross_id](hidden_states).view(
                    -1, self.num_key_value_heads, self.head_dim
                )
                value_states = self.v_proj[cross_id](hidden_states).view(
                    -1, self.num_key_value_heads, self.head_dim
                )
                cross_attention_states[cross_id] = (key_states, value_states)

        if any(state is None for state in cross_attention_states):
            raise RuntimeError(
                "Not all chunk cross-attention states were produced; check `cross_attention_layers` "
                "against the chunk layer count."
            )

        return cross_attention_states


class CombLlamaTextModel(CombLlamaPreTrainedModel):
    config_class = CombLlamaConfig
    base_model_prefix = "language_model.model"

    def __init__(self, config: CombLlamaConfig):
        super().__init__(config)
        text_config = config.get_text_config()
        self.text_config = text_config
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.cross_attention_layers = config.cross_attention_layers
        self._cross_layer_map = {idx: layer_id for layer_id, idx in enumerate(self.cross_attention_layers)}

        num_self_attn_layers = config.num_hidden_layers - len(config.cross_attention_layers)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(text_config, layer_idx) for layer_idx in range(num_self_attn_layers)]
        )
        self.cross_layers = nn.ModuleList(
            [
                CombLlamaCrossAttentionDecoderLayer(text_config, layer_idx)
                for layer_idx in self.cross_attention_layers
            ]
        )
        self.norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _self_attn_forward(
        self,
        decoder_layer: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        decode_cache: Optional[CombLlamaDecodeCache] = None,
        cache_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        num_heads = self.text_config.num_attention_heads
        num_key_value_heads = self.text_config.num_key_value_heads
        head_dim = decoder_layer.self_attn.head_dim

        q = decoder_layer.self_attn.q_proj(hidden_states).view(-1, num_heads, head_dim)
        k = decoder_layer.self_attn.k_proj(hidden_states).view(-1, num_key_value_heads, head_dim)
        v = decoder_layer.self_attn.v_proj(hidden_states).view(-1, num_key_value_heads, head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q, k = q.squeeze(0), k.squeeze(0)

        if decode_cache is not None:
            if cache_layer_idx is None:
                raise ValueError("`cache_layer_idx` is required when writing a decode cache.")
            if k.shape[0] > decode_cache.max_cache_len:
                raise ValueError(
                    f"Prefill length {k.shape[0]} exceeds max cache length {decode_cache.max_cache_len}."
                )
            k_cache, v_cache = decode_cache.past_key_values[cache_layer_idx]
            k_cache[:, : k.shape[0]].copy_(k.unsqueeze(0))
            v_cache[:, : v.shape[0]].copy_(v.unsqueeze(0))

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            dropout_p=0.0 if not self.training else self.text_config.attention_dropout,
            softmax_scale=decoder_layer.self_attn.scaling,
            causal=True,
        )
        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        hidden_states = residual + decoder_layer.self_attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        return residual + hidden_states

    def _self_attn_decode_forward(
        self,
        decoder_layer: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        decode_cache: CombLlamaDecodeCache,
        cache_layer_idx: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        num_heads = self.text_config.num_attention_heads
        num_key_value_heads = self.text_config.num_key_value_heads
        head_dim = decoder_layer.self_attn.head_dim

        q = decoder_layer.self_attn.q_proj(hidden_states).view(1, 1, num_heads, head_dim)
        k = decoder_layer.self_attn.k_proj(hidden_states).view(1, 1, num_key_value_heads, head_dim)
        v = decoder_layer.self_attn.v_proj(hidden_states).view(1, 1, num_key_value_heads, head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        k_cache, v_cache = decode_cache.past_key_values[cache_layer_idx]
        attn_output = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=decode_cache.cache_seqlens,
            softmax_scale=decoder_layer.self_attn.scaling,
            causal=True,
        )
        attn_output = attn_output.view(1, 1, -1)
        hidden_states = residual + decoder_layer.self_attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        return residual + hidden_states

    def _new_decode_cache(
        self,
        prompt_len: int,
        max_new_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CombLlamaDecodeCache:
        max_cache_len = prompt_len + max_new_tokens
        if max_cache_len <= prompt_len:
            raise ValueError("`max_new_tokens` must be positive for generation prefill.")

        num_key_value_heads = self.text_config.num_key_value_heads
        head_dim = self.text_config.hidden_size // self.text_config.num_attention_heads
        past_key_values = []
        for _ in self.layers:
            k_cache = torch.empty(
                1,
                max_cache_len,
                num_key_value_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache = torch.empty_like(k_cache)
            past_key_values.append((k_cache, v_cache))

        cache_seqlens = torch.tensor([prompt_len], device=device, dtype=torch.int32)
        return CombLlamaDecodeCache(
            past_key_values=past_key_values,
            cache_seqlens=cache_seqlens,
            max_cache_len=max_cache_len,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        cross_attention_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        decode_cache: Optional[CombLlamaDecodeCache] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        position_ids = _ensure_packed_2d("position_ids", _require("position_ids", position_ids))
        cu_seqlens_q = _require("cu_seqlens_q", cu_seqlens_q)
        max_seqlen_q = _as_int(_require("max_seqlen_q", max_seqlen_q))
        cross_attention_states = _require("cross_attention_states", cross_attention_states)
        cu_seqlens_k = _require("cu_seqlens_k", cu_seqlens_k)
        max_seqlen_k = _as_int(_require("max_seqlen_k", max_seqlen_k))

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            input_ids = _ensure_packed_2d("input_ids", input_ids)
            inputs_embeds = self.embed_tokens(input_ids.long())

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            cross_layer_id = self._cross_layer_map.get(idx)
            if cross_layer_id is not None:
                hidden_states = self.cross_layers[cross_layer_id](
                    hidden_states=hidden_states,
                    cross_attention_states=cross_attention_states[cross_layer_id],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )

            if self.gradient_checkpointing and self.training and decode_cache is None:
                hidden_states = self._gradient_checkpointing_func(
                    self._self_attn_forward,
                    decoder_layer,
                    hidden_states,
                    position_embeddings,
                    cu_seqlens_q,
                    max_seqlen_q,
                )
            else:
                hidden_states = self._self_attn_forward(
                    decoder_layer,
                    hidden_states,
                    position_embeddings,
                    cu_seqlens_q,
                    max_seqlen_q,
                    decode_cache,
                    idx,
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    def prefill(
        self,
        input_ids: torch.LongTensor,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        max_new_tokens: int,
    ) -> Tuple[BaseModelOutput, CombLlamaDecodeCache]:
        input_ids = _ensure_packed_2d("input_ids", input_ids)
        if input_ids.shape[0] != 1:
            raise NotImplementedError("Only batch_size=1 prefill is supported.")
        prompt_len = input_ids.shape[1]
        if prompt_len <= 0:
            raise ValueError("`input_ids` must contain at least one token.")

        device = input_ids.device
        position_ids = torch.arange(prompt_len, device=device, dtype=torch.long).unsqueeze(0)
        cu_seqlens_q = torch.tensor([0, prompt_len], device=device, dtype=torch.int32)
        inputs_embeds = self.embed_tokens(input_ids.long())
        decode_cache = self._new_decode_cache(
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            device=device,
            dtype=inputs_embeds.dtype,
        )
        outputs = self.forward(
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=prompt_len,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            inputs_embeds=inputs_embeds,
            decode_cache=decode_cache,
        )
        return outputs, decode_cache

    def decode(
        self,
        input_ids: torch.LongTensor,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        decode_cache: CombLlamaDecodeCache,
    ) -> BaseModelOutput:
        input_ids = _ensure_packed_2d("input_ids", input_ids)
        if input_ids.shape != (1, 1):
            raise NotImplementedError("Only single-token batch_size=1 decode is supported.")
        if int(decode_cache.cache_seqlens.item()) >= decode_cache.max_cache_len:
            raise ValueError("Decode cache is full.")

        hidden_states = self.embed_tokens(input_ids.long())
        position_ids = decode_cache.cache_seqlens.to(dtype=torch.long).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cu_seqlens_q = torch.tensor([0, 1], device=input_ids.device, dtype=torch.int32)

        for idx, decoder_layer in enumerate(self.layers):
            cross_layer_id = self._cross_layer_map.get(idx)
            if cross_layer_id is not None:
                hidden_states = self.cross_layers[cross_layer_id](
                    hidden_states=hidden_states,
                    cross_attention_states=cross_attention_states[cross_layer_id],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=1,
                    max_seqlen_k=max_seqlen_k,
                )

            hidden_states = self._self_attn_decode_forward(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                decode_cache=decode_cache,
                cache_layer_idx=idx,
            )

        decode_cache.cache_seqlens += 1
        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class CombLlamaForCausalLM(CombLlamaPreTrainedModel):
    config_class = CombLlamaConfig
    base_model_prefix = "language_model"

    def __init__(self, config: CombLlamaConfig):
        super().__init__(config)
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = CombLlamaTextModel._from_config(config)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        cross_attention_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, torch.Tensor):
            logits = self.lm_head(hidden_states[:, logits_to_keep, :]).float()
        elif logits_to_keep > 0:
            logits = self.lm_head(hidden_states[:, -logits_to_keep:, :]).float()
        else:
            logits = self.lm_head(hidden_states).float()

        return CausalLMOutput(
            loss=_packed_loss(logits, labels),
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def prefill(
        self,
        input_ids: torch.LongTensor,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, CombLlamaDecodeCache]:
        outputs, decode_cache = self.model.prefill(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            max_new_tokens=max_new_tokens,
        )
        logits = self.lm_head(outputs.last_hidden_state[:, -1:, :]).float()
        return logits, decode_cache

    def decode(
        self,
        input_ids: torch.LongTensor,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        decode_cache: CombLlamaDecodeCache,
    ) -> Tuple[torch.Tensor, CombLlamaDecodeCache]:
        outputs = self.model.decode(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            decode_cache=decode_cache,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return logits, decode_cache


@auto_docstring
class CombLlamaForConditionalGeneration(CombLlamaPreTrainedModel):
    config_class = CombLlamaConfig

    def _init_chunk_model_from_language_model(self) -> None:
        self.chunk_model.embed_tokens.load_state_dict(self.language_model.model.embed_tokens.state_dict())

        source_layers = self.language_model.model.layers
        if len(source_layers) != len(self.chunk_model.layers):
            raise ValueError(
                f"Cannot initialize chunk model with {len(self.chunk_model.layers)} layers "
                f"from language model with {len(source_layers)} layers."
            )

        for layer_idx, source_layer in enumerate(source_layers):
            self.chunk_model.layers[layer_idx].load_state_dict(source_layer.state_dict(), strict=True)

        for cross_id, source_layer_idx in enumerate(self.config.cross_attention_layers):
            if source_layer_idx >= len(source_layers):
                raise ValueError(
                    f"Cannot initialize chunk cross K/V {cross_id} from language layer {source_layer_idx}; "
                    f"only {len(source_layers)} language self-attention layers are available."
                )

            source_layer = source_layers[source_layer_idx]
            self.chunk_model.k_proj[cross_id].load_state_dict(source_layer.self_attn.k_proj.state_dict())
            self.chunk_model.v_proj[cross_id].load_state_dict(source_layer.self_attn.v_proj.state_dict())

    def __init__(self, config: CombLlamaConfig, from_scratch: bool = False):
        """
        Args:
            from_scratch (`bool`, *optional*, defaults to `False`):
                Initialize the language model and causal chunk stack from the base Llama checkpoint,
                freeze reused weights, and leave text cross-attention plus chunk K/V projections trainable.
        """
        super().__init__(config)
        self.text_config = config.get_text_config()
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.chunk_model = CombLlamaChunkModel._from_config(config)
        self.language_model = CombLlamaForCausalLM._from_config(config)
        self.post_init()

        if from_scratch:
            self.language_model = CombLlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                config=config,
            )
            self._init_chunk_model_from_language_model()

            for param in self.parameters():
                param.requires_grad = False
            for param in self.chunk_model.k_proj.parameters():
                param.requires_grad = True
            for param in self.chunk_model.v_proj.parameters():
                param.requires_grad = True
            for param in self.language_model.model.cross_layers.parameters():
                param.requires_grad = True

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _single_context_metadata(
        self,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, int]:
        if not cross_attention_states:
            raise ValueError("`cross_attention_states` must contain at least one layer.")
        max_seqlen_k = cross_attention_states[0][0].shape[0]
        device = cross_attention_states[0][0].device
        cu_seqlens_k = torch.tensor([0, max_seqlen_k], device=device, dtype=torch.int32)
        return cu_seqlens_k, max_seqlen_k

    def encode_context(
        self,
        chunk_ids: torch.LongTensor,
        position_ids_k: Optional[torch.LongTensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        chunk_ids = _ensure_packed_2d("chunk_ids", chunk_ids)
        if chunk_ids.shape[0] != 1:
            raise NotImplementedError("Only batch_size=1 context encoding is supported.")

        ctx_len = chunk_ids.shape[1]
        if ctx_len <= 0:
            raise ValueError("`chunk_ids` must contain at least one token.")
        device = chunk_ids.device
        if position_ids_k is None:
            position_ids_k = torch.arange(ctx_len, device=device, dtype=torch.long).unsqueeze(0)
        else:
            position_ids_k = _ensure_packed_2d("position_ids_k", position_ids_k).to(device=device)

        cu_seqlens_k = torch.tensor([0, ctx_len], device=device, dtype=torch.int32)
        return self.chunk_model(
            chunk_ids=chunk_ids,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=ctx_len,
            position_ids_k=position_ids_k,
        )

    def prefill(
        self,
        input_ids: torch.LongTensor,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, CombLlamaDecodeCache]:
        cu_seqlens_k, max_seqlen_k = self._single_context_metadata(cross_attention_states)
        return self.language_model.prefill(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            max_new_tokens=max_new_tokens,
        )

    def decode(
        self,
        input_ids: torch.LongTensor,
        cross_attention_states: List[Tuple[torch.Tensor, torch.Tensor]],
        decode_cache: CombLlamaDecodeCache,
    ) -> Tuple[torch.Tensor, CombLlamaDecodeCache]:
        cu_seqlens_k, max_seqlen_k = self._single_context_metadata(cross_attention_states)
        return self.language_model.decode(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            decode_cache=decode_cache,
        )

    @staticmethod
    def _is_eos(token_id: int, eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]]) -> bool:
        if eos_token_id is None:
            return False
        if isinstance(eos_token_id, (list, tuple)):
            return token_id in eos_token_id
        return token_id == eos_token_id

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        chunk_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        max_new_tokens: int = 128,
        eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        temperature: Optional[float] = 0.0,
        **kwargs,
    ) -> torch.LongTensor:
        if "max_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_tokens")
        if temperature not in (None, 0, 0.0):
            raise NotImplementedError("Only greedy decoding with temperature=0 is supported.")
        if kwargs.get("do_sample", False):
            raise NotImplementedError("Sampling is not supported in the benchmark decode path.")
        if kwargs.get("num_beams", 1) != 1:
            raise NotImplementedError("Beam search is not supported in the benchmark decode path.")
        for unsupported in ("top_p", "top_k", "typical_p"):
            if unsupported in kwargs and kwargs[unsupported] is not None:
                raise NotImplementedError(f"`{unsupported}` is not supported in the benchmark decode path.")

        input_ids = _ensure_packed_2d("input_ids", input_ids)
        if input_ids.shape[0] != 1:
            raise NotImplementedError("Only batch_size=1 generation is supported.")
        if max_new_tokens <= 0:
            return input_ids

        if eos_token_id is None:
            eos_token_id = self.text_config.eos_token_id

        was_training = self.training
        self.eval()
        try:
            if cross_attention_states is None:
                if chunk_ids is None:
                    raise ValueError("Either `chunk_ids` or `cross_attention_states` must be provided.")
                cross_attention_states = self.encode_context(chunk_ids)

            logits, decode_cache = self.prefill(
                input_ids=input_ids,
                cross_attention_states=cross_attention_states,
                max_new_tokens=max_new_tokens,
            )

            generated = []
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if self._is_eos(int(next_token.item()), eos_token_id):
                return torch.cat([input_ids, *generated], dim=1)

            for _ in range(1, max_new_tokens):
                logits, decode_cache = self.decode(
                    input_ids=next_token,
                    cross_attention_states=cross_attention_states,
                    decode_cache=decode_cache,
                )
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                if self._is_eos(int(next_token.item()), eos_token_id):
                    break

            return torch.cat([input_ids, *generated], dim=1)
        finally:
            if was_training:
                self.train()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor,
        chunk_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        position_ids_k: torch.LongTensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        cross_attention_states = self.chunk_model(
            chunk_ids=chunk_ids,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            position_ids_k=position_ids_k,
        )

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            cross_attention_states=cross_attention_states,
            inputs_embeds=inputs_embeds,
            labels=labels,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
        )


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = CombLlamaForConditionalGeneration(
        from_scratch=True,
        config=CombLlamaConfig(LlamaConfig.from_pretrained(model_name)),
    )
