# SPDX-License-Identifier: Apache-2.0
"""Frozen 32-layer PIC encoder with a trainable context-attention branch.

This is intentionally separate from :mod:`CombLlama`.  The official Comb
implementation remains available unchanged; this module implements the
controlled architecture used by the next experiment:

* a pretrained, frozen 32-layer Llama chunk encoder;
* a pretrained, frozen 32-layer Llama language model;
* eight trainable context branches inserted before decoder layers
  ``[3, 7, 11, 15, 19, 23, 27, 31]``;
* Flash Attention for the two Llama backbones and explicit SDPA for cross
  attention; and
* an MLP that only receives the context-attention output, so it cannot become
  a query-only shortcut.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.modeling_utils import no_init_weights

from .CombLlama import (
    CombLlamaConfig,
    CombLlamaForCausalLM,
    CombLlamaForConditionalGeneration,
    CombLlamaPreTrainedModel,
    CombLlamaTextModel,
)


DEFAULT_CROSS_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


class CombLlamaFrozen32Config(CombLlamaConfig):
    """Configuration with separate encoder, decoder, and cross-layer counts."""

    model_type = "combllama_frozen32"

    def __init__(
        self,
        text_config: Optional[LlamaConfig] = None,
        encoder_num_hidden_layers: int = 32,
        decoder_num_hidden_layers: Optional[int] = None,
        cross_attention_layers: Optional[List[int]] = None,
        encoder_attn_implementation: str = "flash_attention_2",
        decoder_attn_implementation: str = "flash_attention_2",
        cross_attn_implementation: str = "sdpa",
        context_gate_init: float = 1e-3,
        **kwargs,
    ) -> None:
        # ``CombLlamaConfig.to_dict`` serializes this derived cache-slot count.
        # Consume it here so loading a saved Frozen32 config does not pass the
        # same keyword twice to the parent constructor.
        serialized_cache_layer_count = kwargs.pop("num_hidden_layers", None)
        if text_config is None:
            text_config = LlamaConfig()
        elif isinstance(text_config, dict):
            text_config = LlamaConfig(**text_config)
        else:
            # CombLlamaConfig mutates text_config.num_hidden_layers to match
            # its cache-slot count.  Never leak that implementation detail
            # back into a caller's backbone configuration.
            text_config = copy.deepcopy(text_config)

        if cross_attention_layers is None:
            cross_attention_layers = list(DEFAULT_CROSS_LAYERS)
        if decoder_num_hidden_layers is None:
            decoder_num_hidden_layers = int(text_config.num_hidden_layers)

        if sorted(cross_attention_layers) != cross_attention_layers:
            raise ValueError("cross_attention_layers must be sorted")
        if len(set(cross_attention_layers)) != len(cross_attention_layers):
            raise ValueError("cross_attention_layers must be unique")
        if not cross_attention_layers or cross_attention_layers[-1] >= decoder_num_hidden_layers:
            raise ValueError("cross_attention_layers must index decoder layers")
        if cross_attn_implementation != "sdpa":
            raise ValueError("the initial frozen32 implementation requires SDPA cross-attention")

        # CombLlamaTextModel reserves the trailing cache slots for cross K/V.
        cache_layer_count = decoder_num_hidden_layers + len(cross_attention_layers)
        if (
            serialized_cache_layer_count is not None
            and int(serialized_cache_layer_count) != cache_layer_count
        ):
            raise ValueError(
                "serialized num_hidden_layers does not match Frozen32 cache slots: "
                f"serialized={serialized_cache_layer_count}, expected={cache_layer_count}"
            )
        super().__init__(
            text_config=text_config,
            num_hidden_layers=cache_layer_count,
            cross_attention_layers=cross_attention_layers,
            **kwargs,
        )
        self.encoder_num_hidden_layers = int(encoder_num_hidden_layers)
        self.decoder_num_hidden_layers = int(decoder_num_hidden_layers)
        self.encoder_attn_implementation = encoder_attn_implementation
        self.decoder_attn_implementation = decoder_attn_implementation
        self.cross_attn_implementation = cross_attn_implementation
        self.context_gate_init = float(context_gate_init)
        # CombLlamaTextModel owns the causal-mask construction and reads the
        # composite config, while each LlamaDecoderLayer reads text_config.
        # They must advertise the same backend or an eager decoder can receive
        # no causal mask at all.
        self._attn_implementation = decoder_attn_implementation
        self.text_config._attn_implementation = decoder_attn_implementation


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand GQA K/V heads without changing the canonical PIC layout."""

    if groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(groups, dim=1)


class Frozen32PICCrossAttention(nn.Module):
    """Non-causal cross attention with explicit PyTorch SDPA dispatch."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def project_pic(
        self, encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return canonical PIC tensors shaped ``[B, Lc, Hkv, D]``."""

        batch, context_length, _ = encoder_hidden_states.shape
        key = self.k_proj(encoder_hidden_states).view(
            batch, context_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(encoder_hidden_states).view(
            batch, context_length, self.num_key_value_heads, self.head_dim
        )
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states,
        past_key_value=None,
        attention_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, query_length, self.num_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)

        if cross_attention_states is not None:
            if isinstance(cross_attention_states, tuple):
                key, value = cross_attention_states
            else:
                key, value = self.project_pic(cross_attention_states)
            key = self.k_norm(key).transpose(1, 2)
            value = value.transpose(1, 2)
            if past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) == 0:
                key, value = past_key_value.update(key, value, self.layer_idx)
        elif past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) > 0:
            key = past_key_value.layers[self.layer_idx].keys
            value = past_key_value.layers[self.layer_idx].values
        else:
            raise ValueError("cross attention requires PIC states or a populated PIC cache")

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        context = context.transpose(1, 2).reshape(batch, query_length, self.hidden_size)
        return self.o_proj(context)


class Frozen32PICContextLayer(nn.Module):
    """Gated context branch whose MLP cannot operate without context."""

    def __init__(self, config: LlamaConfig, layer_idx: int, gate_init: float) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = Frozen32PICCrossAttention(config, layer_idx)
        self.context_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)
        self.context_gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states,
        cross_attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        **kwargs,
    ) -> torch.Tensor:
        context = self.cross_attn(
            self.input_layernorm(hidden_states),
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            attention_mask=cross_attention_mask,
            **kwargs,
        )
        context = context + self.mlp(self.context_layernorm(context))
        return hidden_states + self.context_gate.tanh().to(context.dtype) * context


class Frozen32PICChunkModel(CombLlamaPreTrainedModel):
    """Frozen pretrained Llama encoder returning eight intermediate states."""

    base_model_prefix = "chunk_model"

    def __init__(self, config: CombLlamaFrozen32Config) -> None:
        encoder_config = copy.deepcopy(config.get_text_config())
        encoder_config.num_hidden_layers = config.encoder_num_hidden_layers
        encoder_config._attn_implementation = config.encoder_attn_implementation
        super().__init__(encoder_config)
        self.tap_layers = list(config.cross_attention_layers)
        self.model = LlamaModel(encoder_config)

    def forward(
        self, chunk_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        # The official wrapper supplies the cross mask after converting it to an
        # additive 4-D mask.  Recover the original 2-D encoder padding mask.
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = (attention_mask[:, 0, 0, :] == 0).to(torch.long)
        with torch.no_grad():
            outputs = self.model(
                input_ids=chunk_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return [outputs.hidden_states[index + 1] for index in self.tap_layers]


class Frozen32PICTextModel(CombLlamaTextModel):
    """The frozen 32-layer decoder with trainable context branches."""

    def __init__(self, config: CombLlamaFrozen32Config) -> None:
        CombLlamaPreTrainedModel.__init__(self, config)
        text_config = config.get_text_config()
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = nn.Embedding(
            text_config.vocab_size, text_config.hidden_size, self.padding_idx
        )
        self.cross_attention_layers = list(config.cross_attention_layers)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(text_config, layer_idx)
            for layer_idx in range(config.decoder_num_hidden_layers)
        )
        cache_offset = config.decoder_num_hidden_layers
        self.cross_layers = nn.ModuleList(
            Frozen32PICContextLayer(
                text_config,
                layer_idx=cache_offset + cross_id,
                gate_init=config.context_gate_init,
            )
            for cross_id, _ in enumerate(self.cross_attention_layers)
        )
        self.norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.gradient_checkpointing = False
        self.post_init()
        for layer in self.cross_layers:
            layer.context_gate.data.fill_(config.context_gate_init)


class Frozen32PICForCausalLM(CombLlamaForCausalLM):
    def __init__(self, config: CombLlamaFrozen32Config) -> None:
        CombLlamaPreTrainedModel.__init__(self, config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = Frozen32PICTextModel(config)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()


class CombLlamaFrozen32ForConditionalGeneration(CombLlamaForConditionalGeneration):
    """Conditional generation model for the frozen32 PIC experiment."""

    config_class = CombLlamaFrozen32Config

    def __init__(self, config: CombLlamaFrozen32Config) -> None:
        CombLlamaPreTrainedModel.__init__(self, config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.chunk_model = Frozen32PICChunkModel(config)
        self.language_model = Frozen32PICForCausalLM(config)
        self.post_init()
        self._set_frozen32_trainability()

    def _set_frozen32_trainability(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.language_model.model.cross_layers.parameters():
            parameter.requires_grad = True

    @torch.no_grad()
    def initialize_from_llama(self, backbone: LlamaForCausalLM) -> None:
        """Copy both frozen backbones and initialize every context branch."""

        expected = self.config.decoder_num_hidden_layers
        if len(backbone.model.layers) != expected:
            raise ValueError(
                f"expected a {expected}-layer Llama backbone, got {len(backbone.model.layers)}"
            )
        self.chunk_model.model.load_state_dict(backbone.model.state_dict(), strict=True)
        self.language_model.model.embed_tokens.load_state_dict(
            backbone.model.embed_tokens.state_dict()
        )
        self.language_model.model.norm.load_state_dict(backbone.model.norm.state_dict())
        self.language_model.lm_head.load_state_dict(backbone.lm_head.state_dict())
        for target, source in zip(self.language_model.model.layers, backbone.model.layers):
            target.load_state_dict(source.state_dict(), strict=True)

        for decoder_index, context_layer in zip(
            self.config.cross_attention_layers,
            self.language_model.model.cross_layers,
        ):
            source = backbone.model.layers[decoder_index]
            context_layer.input_layernorm.load_state_dict(source.input_layernorm.state_dict())
            context_layer.context_layernorm.load_state_dict(
                source.post_attention_layernorm.state_dict()
            )
            context_layer.mlp.load_state_dict(source.mlp.state_dict(), strict=True)
            context_layer.cross_attn.q_proj.load_state_dict(
                source.self_attn.q_proj.state_dict(), strict=True
            )
            context_layer.cross_attn.k_proj.load_state_dict(
                source.self_attn.k_proj.state_dict(), strict=True
            )
            context_layer.cross_attn.v_proj.load_state_dict(
                source.self_attn.v_proj.state_dict(), strict=True
            )
            context_layer.cross_attn.o_proj.load_state_dict(
                source.self_attn.o_proj.state_dict(), strict=True
            )
            context_layer.cross_attn.q_norm.weight.fill_(1.0)
            context_layer.cross_attn.k_norm.weight.fill_(1.0)
            context_layer.context_gate.fill_(self.config.context_gate_init)
        self._set_frozen32_trainability()

    @classmethod
    def from_llama_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: Optional[CombLlamaFrozen32Config] = None,
        **kwargs,
    ) -> "CombLlamaFrozen32ForConditionalGeneration":
        backbone = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if config is None:
            config = CombLlamaFrozen32Config(copy.deepcopy(backbone.config))
        # Construct the second backbone directly in the requested storage
        # dtype.  Building this 17B-parameter composite in FP32 and casting it
        # afterwards would transiently require roughly twice the host memory.
        requested_dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        previous_dtype = torch.get_default_dtype()
        if requested_dtype is not None:
            torch.set_default_dtype(requested_dtype)
        try:
            # Every parameter is immediately overwritten from the pretrained
            # backbone below, including all cross-branch initial values.  Skip
            # an otherwise very expensive random initialization pass.
            with no_init_weights():
                model = cls(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        model.initialize_from_llama(backbone)
        del backbone
        return model

    def build_pic_cache(
        self,
        chunk_ids: torch.Tensor,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Build reusable canonical K/V tensors for one independently encoded chunk."""

        hidden_states = self.chunk_model(chunk_ids, cross_attention_mask)
        return [
            layer.cross_attn.project_pic(hidden)
            for layer, hidden in zip(self.language_model.model.cross_layers, hidden_states)
        ]

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


__all__ = [
    "CombLlamaFrozen32Config",
    "CombLlamaFrozen32ForConditionalGeneration",
    "Frozen32PICContextLayer",
    "Frozen32PICCrossAttention",
]
