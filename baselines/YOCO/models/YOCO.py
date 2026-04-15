"""YOCO-Llama baseline.

Stage 2 implements the first runnable 16+16 forward path:

- the self-decoder uses sliding-window attention (SWA)
- the cross-decoder attends to the final self-decoder memory states
- cache-aware decoding is deferred to Stage 3
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from flash_attn import flash_attn_varlen_func


def _compute_loss(logits, labels, shift_labels, vocab_size):
    if labels is None and shift_labels is None:
        return None
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, vocab_size), shift_labels.view(-1))


def _slice_logits(lm_head, hidden_states, logits_to_keep):
    if isinstance(logits_to_keep, torch.Tensor):
        logits = lm_head(hidden_states[:, logits_to_keep, :])
    elif logits_to_keep > 0:
        logits = lm_head(hidden_states[:, -logits_to_keep:, :])
    else:
        logits = lm_head(hidden_states)
    return logits.float()


def _apply_rotary_to_single(
    x: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = position_embeddings
    x_out, _ = apply_rotary_pos_emb(x.unsqueeze(0), x.unsqueeze(0), cos, sin, unsqueeze_dim=2)
    return x_out.squeeze(0)


class YOCOConfig(PretrainedConfig):
    """Configuration for the YOCO-Llama baseline."""

    model_type = "yoco_llama"
    sub_configs = {"text_config": LlamaConfig}

    def __init__(
        self,
        text_config: Optional[LlamaConfig] = None,
        num_self_decoder_layers: int = 16,
        num_cross_decoder_layers: int = 16,
        pad_token_id: Optional[int] = 128004,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            self.text_config = LlamaConfig()
        elif isinstance(text_config, dict):
            self.text_config = LlamaConfig(**text_config)
        elif isinstance(text_config, LlamaConfig):
            self.text_config = text_config
        else:
            raise TypeError("text_config must be None, dict, or LlamaConfig")

        self.num_self_decoder_layers = num_self_decoder_layers
        self.num_cross_decoder_layers = num_cross_decoder_layers
        self.num_hidden_layers = num_self_decoder_layers + num_cross_decoder_layers

        if self.num_hidden_layers != self.text_config.num_hidden_layers:
            raise ValueError(
                "YOCO layer split must match text_config.num_hidden_layers: "
                f"{self.num_hidden_layers} != {self.text_config.num_hidden_layers}"
            )

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class YOCODynamicCache:
    """Minimal cache container for Stage 3 YOCO decoding."""

    def __init__(self, num_self_decoder_layers: int, window_size: int):
        self.num_self_decoder_layers = num_self_decoder_layers
        self.window_size = window_size
        self.self_keys = [None] * num_self_decoder_layers
        self.self_values = [None] * num_self_decoder_layers
        self.memory_states = None
        self.memory_position_ids = None

    def get_self_layer_cache(self, layer_idx: int):
        return self.self_keys[layer_idx], self.self_values[layer_idx]

    def update_self_layer_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        self.self_keys[layer_idx] = key_states
        self.self_values[layer_idx] = value_states

    def append_memory(self, memory_states: torch.Tensor, position_ids: torch.Tensor):
        memory_states = memory_states.squeeze(0)
        position_ids = position_ids.squeeze(0)
        if self.memory_states is None:
            self.memory_states = memory_states
            self.memory_position_ids = position_ids
        else:
            self.memory_states = torch.cat([self.memory_states, memory_states], dim=0)
            self.memory_position_ids = torch.cat([self.memory_position_ids, position_ids], dim=0)

    def get_memory(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.memory_states, self.memory_position_ids

    def get_seq_length(self) -> int:
        if self.memory_states is None:
            return 0
        return self.memory_states.shape[0]


@auto_docstring
class YOCOPreTrainedModel(PreTrainedModel):
    """Base class for YOCO models."""

    config_class = YOCOConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "YOCOSelfDecoder",
        "YOCOSelfDecoderLayer",
        "YOCOCrossDecoder",
        "YOCOCrossDecoderLayer",
    ]
    _supports_flash_attn = True

    def _init_weights(self, module):
        std = getattr(
            self.config,
            "initializer_range",
            self.config.get_text_config().initializer_range,
        )

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


class YOCOSlidingWindowAttention(nn.Module):
    """Sliding-window self-attention used by the YOCO self-decoder."""

    def __init__(self, config: LlamaConfig, window_size: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.window_size = window_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        q = _apply_rotary_to_single(q, position_embeddings)
        k = _apply_rotary_to_single(k, position_embeddings)

        if past_key_value is None:
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=True,
                window_size=(self.window_size - 1, 0),
            )
            key_states = k
            value_states = v
        else:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, k], dim=0)
            value_states = torch.cat([past_v, v], dim=0)
            cu_seqlens_k = torch.tensor(
                [0, key_states.shape[0]], dtype=torch.int32, device=hidden_states.device
            )
            attn_output = flash_attn_varlen_func(
                q,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=key_states.shape[0],
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=False,
            )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        next_past = None
        if use_cache:
            next_past = (
                key_states[-self.window_size :].contiguous(),
                value_states[-self.window_size :].contiguous(),
            )
        return self.o_proj(attn_output), next_past


class YOCOSelfDecoderLayer(nn.Module):
    """A single self-decoder block using sliding-window attention."""

    def __init__(self, config: LlamaConfig, window_size: int) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = YOCOSlidingWindowAttention(config, window_size=window_size)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, next_past = self.self_attn(
            hidden_states,
            position_embeddings,
            cu_seqlens_q,
            max_seqlen_q,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, next_past


class YOCOCrossAttention(nn.Module):
    """Cross-attention from cross-decoder queries to self-decoder memory."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        memory_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        causal: bool,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(memory_states).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(memory_states).view(-1, self.num_key_value_heads, self.head_dim)

        q = _apply_rotary_to_single(q, query_position_embeddings)
        k = _apply_rotary_to_single(k, memory_position_embeddings)

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=causal,
        )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        return self.o_proj(attn_output)


class YOCOCrossDecoderLayer(nn.Module):
    """Cross-decoder layer for the YOCO-Llama baseline."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = YOCOCrossAttention(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        memory_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        causal: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states,
            memory_states,
            query_position_embeddings,
            memory_position_embeddings,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            causal,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class YOCOSelfDecoder(YOCOPreTrainedModel):
    """Self-decoder block group for the YOCO-Llama baseline."""

    config_class = YOCOConfig
    base_model_prefix = "model.self_decoder"

    def __init__(self, config: YOCOConfig):
        text_config = config.get_text_config()
        super().__init__(config)
        self.window_size = getattr(text_config, "sliding_window", 512)
        self.layers = nn.ModuleList(
            [YOCOSelfDecoderLayer(text_config, self.window_size) for _ in range(config.num_self_decoder_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        past_key_values: Optional[YOCODynamicCache] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        all_hidden_states = () if output_hidden_states else None

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_past = None
            if past_key_values is not None:
                past_k, past_v = past_key_values.get_self_layer_cache(layer_idx)
                if past_k is not None:
                    layer_past = (past_k, past_v)
            hidden_states, next_past = layer(
                hidden_states,
                position_embeddings,
                cu_seqlens_q,
                max_seqlen_q,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and past_key_values is not None and next_past is not None:
                past_key_values.update_self_layer_cache(layer_idx, *next_past)

        return hidden_states, all_hidden_states


class YOCOCrossDecoder(YOCOPreTrainedModel):
    """Cross-decoder block group for the YOCO-Llama baseline."""

    config_class = YOCOConfig
    base_model_prefix = "model.cross_decoder"

    def __init__(self, config: YOCOConfig):
        text_config = config.get_text_config()
        super().__init__(config)
        start_idx = config.num_self_decoder_layers
        end_idx = start_idx + config.num_cross_decoder_layers
        self.layers = nn.ModuleList(
            [YOCOCrossDecoderLayer(text_config, layer_idx) for layer_idx in range(start_idx, end_idx)]
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        memory_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        causal: bool,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(
                hidden_states,
                memory_states,
                query_position_embeddings,
                memory_position_embeddings,
                cu_seqlens_q,
                max_seqlen_q,
                cu_seqlens_k,
                max_seqlen_k,
                causal,
            )

        return hidden_states, all_hidden_states


class YOCOTextModel(YOCOPreTrainedModel):
    """YOCO text model skeleton with separate self-decoder and cross-decoder."""

    config_class = YOCOConfig
    base_model_prefix = "model"

    def __init__(self, config: YOCOConfig):
        text_config = config.get_text_config()
        super().__init__(config)
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = nn.Embedding(
            text_config.vocab_size,
            text_config.hidden_size,
            self.padding_idx,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.self_decoder = YOCOSelfDecoder._from_config(config)
        self.cross_decoder = YOCOCrossDecoder._from_config(config)
        self.norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            raise ValueError("position_ids must be provided for YOCO packed decoding.")
        if cu_seqlens_q is None or max_seqlen_q is None:
            raise ValueError("cu_seqlens_q and max_seqlen_q must be provided.")

        if use_cache:
            if cu_seqlens_q.numel() != 2:
                raise NotImplementedError("Stage 3 cache path currently supports one sequence per batch.")
            if past_key_values is None:
                past_key_values = YOCODynamicCache(
                    self.config.num_self_decoder_layers,
                    self.self_decoder.window_size,
                )
            elif not isinstance(past_key_values, YOCODynamicCache):
                raise TypeError("past_key_values must be a YOCODynamicCache for YOCO cache mode.")

        hidden_states = inputs_embeds
        query_position_embeddings = self.rotary_emb(hidden_states, position_ids)

        self_hidden_states, self_all_hidden_states = self.self_decoder(
            hidden_states,
            query_position_embeddings,
            cu_seqlens_q,
            max_seqlen_q,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if use_cache:
            past_key_values.append_memory(self_hidden_states, position_ids)
            memory_states, memory_position_ids = past_key_values.get_memory()
            memory_states = memory_states.unsqueeze(0)
            memory_position_ids = memory_position_ids.unsqueeze(0)
            memory_position_embeddings = self.rotary_emb(memory_states, memory_position_ids)
            cu_seqlens_k = torch.tensor(
                [0, memory_states.shape[1]], dtype=torch.int32, device=hidden_states.device
            )
            max_seqlen_k = memory_states.shape[1]
            causal = False
        else:
            memory_states = self_hidden_states
            memory_position_embeddings = query_position_embeddings
            cu_seqlens_k = cu_seqlens_q
            max_seqlen_k = max_seqlen_q
            causal = True

        cross_hidden_states, cross_all_hidden_states = self.cross_decoder(
            self_hidden_states,
            memory_states,
            query_position_embeddings,
            memory_position_embeddings,
            cu_seqlens_q,
            max_seqlen_q,
            cu_seqlens_k,
            max_seqlen_k,
            causal,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self.norm(cross_hidden_states)

        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = ()
            if self_all_hidden_states is not None:
                all_hidden_states += self_all_hidden_states
            all_hidden_states += (self_hidden_states,)
            if cross_all_hidden_states is not None:
                all_hidden_states += cross_all_hidden_states
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class YOCOForCausalLM(YOCOPreTrainedModel, GenerationMixin):
    """Causal LM wrapper for the YOCO-Llama baseline."""

    config_class = YOCOConfig
    _supports_static_cache = True
    base_model_prefix = "language_model"

    def __init__(self, config: YOCOConfig):
        text_config = config.get_text_config()
        super().__init__(config)
        self.text_config = text_config
        self.vocab_size = text_config.vocab_size
        self.model = YOCOTextModel._from_config(config)
        self.lm_head = nn.Linear(text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        logits = _slice_logits(self.lm_head, outputs.last_hidden_state, logits_to_keep)
        loss = _compute_loss(logits, labels, shift_labels, self.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
