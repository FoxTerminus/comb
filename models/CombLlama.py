"""CombLlama: a hybrid architecture combining a chunk encoder with a Llama decoder
via cross-attention, designed to compress KV cache for long-context inference."""

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding, LlamaDecoderLayer, apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from typing import List, Optional, Tuple, Union
from flash_attn import flash_attn_varlen_func

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_loss(logits, labels, shift_labels, vocab_size):
    """Compute cross-entropy loss from logits and (shifted) labels.

    Args:
        logits: Model logits ``[batch, seq_len, vocab_size]``.
        labels: Standard labels (presence triggers loss computation).
        shift_labels: Pre-shifted labels ``[batch, seq_len]``.
        vocab_size: Vocabulary size for reshaping.

    Returns:
        Scalar loss tensor, or ``None`` if neither labels nor shift_labels is provided.
    """
    if labels is None and shift_labels is None:
        return None
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, vocab_size), shift_labels.view(-1))


def _slice_logits(lm_head, hidden_states, logits_to_keep):
    """Compute logits, optionally slicing hidden states for memory efficiency.

    Args:
        lm_head: Language model head (linear layer).
        hidden_states: Last hidden states ``[batch, seq_len, hidden]``.
        logits_to_keep: If ``int > 0``, keep only the last N positions.
            If ``0``, keep all. If ``Tensor``, use as positional index.

    Returns:
        Logits tensor cast to float32.
    """
    if isinstance(logits_to_keep, torch.Tensor):
        logits = lm_head(hidden_states[:, logits_to_keep, :])
    elif logits_to_keep > 0:
        logits = lm_head(hidden_states[:, -logits_to_keep:, :])
    else:
        logits = lm_head(hidden_states)
    return logits.float()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CombLlamaConfig(PretrainedConfig):
    r"""Configuration for :class:`CombLlamaForConditionalGeneration`.

    Defines the architecture of a CombLlama model, which augments a Llama backbone
    with cross-attention layers that attend to compressed chunk representations.

    Args:
        text_config (``LlamaConfig`` or ``dict``, *optional*):
            Configuration of the Llama text backbone. Defaults to ``LlamaConfig()``.
        chunk_token_index (``int``, *optional*, defaults to 128255):
            Token index reserved for chunk encoding.
        num_hidden_layers (``int``, *optional*, defaults to 40):
            Total number of hidden layers (backbone + chunk model combined).
        cross_attention_layers (``list[int]``, *optional*):
            Backbone layer indices at which to insert cross-attention.
            Defaults to ``[3, 7, 11, 15, 19, 23, 27, 31]``.
        pad_token_id (``int``, *optional*, defaults to 128004):
            Padding token id.
        tie_word_embeddings (``bool``, *optional*, defaults to ``False``):
            Whether to tie input and output embedding weights.

    Example:
    ```python
    >>> from transformers import LlamaConfig
    >>> from models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
    >>> config = CombLlamaConfig(LlamaConfig())
    >>> model = CombLlamaForConditionalGeneration(config)
    ```
    """

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

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Attention & Layer Modules
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Cross-attention from decoder queries to chunk encoder key/values.

    Uses ``flash_attn_varlen_func`` for variable-length sequence support
    (continuous batching). The KV cache layer index is offset by
    ``config.num_hidden_layers`` to avoid collisions with self-attention cache.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        # Offset layer index to avoid KV cache collision with self-attention layers
        self.layer_idx = layer_idx + config.num_hidden_layers
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        """Compute cross-attention between decoder queries and chunk key/values.

        Args:
            hidden_states: Decoder hidden states ``[1, total_q_len, hidden]``.
            cross_attention_states: ``(key, value)`` from the chunk encoder,
                each ``[total_chunk_len, num_kv_heads, head_dim]``.
            past_key_value: Optional KV cache for autoregressive inference.
            cache_position: Token positions for cache updates.
            cu_seqlens_q: Cumulative query sequence lengths ``[batch + 1]``.
            cu_seqlens_k: Cumulative key/value sequence lengths ``[batch + 1]``.
            max_seqlen_q: Maximum query sequence length in the batch.
            max_seqlen_k: Maximum key/value sequence length in the batch.

        Returns:
            ``(attention_output, past_key_value)`` tuple.
        """
        query_states = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states, value_states = cross_attention_states
            key_states = self.k_norm(key_states)
            if past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) == 0:
                k_4d = key_states.transpose(0, 1).unsqueeze(0)
                v_4d = value_states.transpose(0, 1).unsqueeze(0)
                k_4d, v_4d = past_key_value.update(
                    k_4d, v_4d, self.layer_idx, {"cache_position": cache_position},
                )
                key_states = k_4d.squeeze(0).transpose(0, 1)
                value_states = v_4d.squeeze(0).transpose(0, 1)
        elif cache_position is not None and cache_position[0] != 0:
            key_states = past_key_value.layers[self.layer_idx].keys.squeeze(0).transpose(0, 1)
            value_states = past_key_value.layers[self.layer_idx].values.squeeze(0).transpose(0, 1)
        else:
            raise ValueError(
                "Cross-attention requires either `cross_attention_states` or cached key/values."
            )

        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=False,
            **kwargs,
        )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value


class ChunkVarlenAttention(nn.Module):
    """Bidirectional self-attention for the chunk encoder.

    Processes variable-length chunks without padding via cumulative sequence
    length indexing (``flash_attn_varlen_func``). Uses RoPE for positional encoding.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run bidirectional self-attention over packed variable-length chunks.

        Args:
            hidden_states: Packed chunk embeddings ``[total_chunk_len, hidden]``.
            cos: Cosine RoPE embeddings ``[1, total_chunk_len, head_dim]``.
            sin: Sine RoPE embeddings ``[1, total_chunk_len, head_dim]``.
            cu_seqlens: Cumulative chunk lengths ``[num_chunks + 1]``.
            max_seqlen: Maximum chunk length in the batch.

        Returns:
            Attention output ``[total_chunk_len, hidden]``.
        """
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q, k = q.squeeze(0), k.squeeze(0)

        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=False,
        )

        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class ChunkVarlenLayer(nn.Module):
    """Transformer block for the chunk encoder.

    Pre-norm residual architecture with bidirectional self-attention
    (:class:`ChunkVarlenAttention`) and SwiGLU MLP, supporting variable-length
    chunks via flash attention.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ChunkVarlenAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Pre-norm forward pass with residual connections.

        Args:
            hidden_states: Input embeddings ``[total_chunk_len, hidden]``.
            cos: Cosine RoPE embeddings.
            sin: Sine RoPE embeddings.
            cu_seqlens: Cumulative chunk lengths ``[num_chunks + 1]``.
            max_seqlen: Maximum chunk length.

        Returns:
            Output embeddings ``[total_chunk_len, hidden]``.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CombLlamaCrossAttentionDecoderLayer(nn.Module):
    """Cross-attention decoder layer with tanh-gated residual connections.

    Both gates are initialized to zero so the cross-attention contribution
    starts at zero, preserving the pretrained backbone's behavior at the
    beginning of training.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(config, layer_idx=layer_idx)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        past_key_value: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor]:
        """Forward pass with tanh-gated cross-attention and MLP.

        Args:
            hidden_states: Decoder hidden states ``[1, total_q_len, hidden]``.
            cross_attention_states: ``(key, value)`` from chunk encoder.
            cu_seqlens_q: Cumulative query lengths ``[batch + 1]``.
            cu_seqlens_k: Cumulative key/value lengths ``[batch + 1]``.
            max_seqlen_q: Maximum query length.
            max_seqlen_k: Maximum key/value length.
            past_key_value: Optional KV cache.
            cache_position: Token positions for cache updates.

        Returns:
            Tuple of ``(hidden_states,)``.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        return (hidden_states,)


# ---------------------------------------------------------------------------
# Pretrained Base
# ---------------------------------------------------------------------------

@auto_docstring
class CombLlamaPreTrainedModel(PreTrainedModel):
    """Base class for all CombLlama models.

    Provides weight initialization and module-splitting configuration
    for model parallelism.
    """

    config_class = CombLlamaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "CombLlamaChunkModel",
        "CombLlamaCrossAttentionDecoderLayer",
        "LlamaDecoderLayer",
    ]
    _supports_flash_attn = True

    def _init_weights(self, module):
        std = getattr(
            self.config, "initializer_range",
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
        elif isinstance(module, CombLlamaCrossAttentionDecoderLayer):
            module.cross_attn_attn_gate.data.zero_()
            module.cross_attn_mlp_gate.data.zero_()


# ---------------------------------------------------------------------------
# Chunk Encoder
# ---------------------------------------------------------------------------

class CombLlamaChunkModel(CombLlamaPreTrainedModel):
    """Chunk encoder that produces per-layer key/value states for cross-attention.

    Encodes variable-length history chunks through *N* transformer layers
    (one per cross-attention insertion point in the decoder). Each layer
    outputs projected K/V states that the corresponding decoder cross-attention
    layer will attend to.
    """

    config_class = CombLlamaConfig
    base_model_prefix = "chunk_model"

    def __init__(self, config: CombLlamaConfig):
        self.num_cross_layers = len(config.cross_attention_layers)
        text_config = config.get_text_config()
        super().__init__(text_config)
        self.hidden_size = text_config.hidden_size
        self.head_dim = self.hidden_size // text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads

        self.embed_tokens = nn.Embedding(text_config.vocab_size, self.hidden_size, text_config.pad_token_id)
        self.layers = nn.ModuleList(
            [ChunkVarlenLayer(text_config) for _ in range(self.num_cross_layers)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.k_proj = nn.ModuleList([
            nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            for _ in range(self.num_cross_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            for _ in range(self.num_cross_layers)
        ])
        self.post_init()

    def forward(
        self,
        chunk_ids: torch.Tensor,
        cu_seqlens_chunk: torch.Tensor,
        max_seqlen_chunk: int,
        position_ids_k: torch.Tensor,
        **kwargs,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Encode chunks and produce per-layer K/V states for cross-attention.

        Args:
            chunk_ids: Packed token ids of all chunks ``[1, total_chunk_len]``.
            cu_seqlens_chunk: Cumulative chunk lengths ``[num_chunks + 1]``.
            max_seqlen_chunk: Maximum individual chunk length in the batch.
            position_ids_k: Position ids for chunks ``[1, total_chunk_len]``.

        Returns:
            List of ``(key_states, value_states)`` tuples, one per cross-attention
            layer. Each tensor has shape ``[total_chunk_len, num_kv_heads, head_dim]``.
        """
        hidden_states = self.embed_tokens(chunk_ids.long()).squeeze(0)   # [total_chunk_len, hidden]
        cos, sin = self.rotary_emb(hidden_states.unsqueeze(0), position_ids_k)

        cross_attention_states = []
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cos, sin, cu_seqlens_chunk, max_seqlen_chunk)

            k = self.k_proj[idx](hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
            v = self.v_proj[idx](hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
            cross_attention_states.append((k, v))

        return cross_attention_states


# ---------------------------------------------------------------------------
# Text Decoder
# ---------------------------------------------------------------------------

class CombLlamaTextModel(CombLlamaPreTrainedModel):
    """Llama decoder augmented with interleaved cross-attention layers.

    Standard Llama causal self-attention layers are interleaved with
    :class:`CombLlamaCrossAttentionDecoderLayer` at configurable positions.
    Both self-attention and cross-attention use ``flash_attn_varlen_func``
    for continuous-batching support.
    """

    config_class = CombLlamaConfig
    base_model_prefix = "language_model.model"

    def __init__(self, config: CombLlamaConfig):
        text_config = config.get_text_config()
        super().__init__(text_config)
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.cross_attention_layers = config.cross_attention_layers

        # Pre-compute mapping: backbone layer index -> cross-attention layer index
        self._cross_layer_map = {idx: i for i, idx in enumerate(self.cross_attention_layers)}

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(text_config, layer_idx)
             for layer_idx in range(text_config.num_hidden_layers)]
        )
        self.cross_layers = nn.ModuleList(
            [CombLlamaCrossAttentionDecoderLayer(text_config, idx)
             for idx in self.cross_attention_layers]
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
    ) -> torch.Tensor:
        """Execute one self-attention + MLP block using ``flash_attn_varlen_func``.

        Extracted as a proper method (rather than an inner closure) so that
        ``decoder_layer`` is passed by value, which is required for correct
        gradient-checkpointing behavior.

        Args:
            decoder_layer: The :class:`LlamaDecoderLayer` to execute.
            hidden_states: Input ``[1, total_len, hidden]``.
            position_embeddings: Precomputed ``(cos, sin)`` tuple.
            cu_seqlens_q: Cumulative query lengths ``[batch + 1]``.
            max_seqlen_q: Maximum query length.

        Returns:
            Output hidden states ``[1, total_len, hidden]``.
        """
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = decoder_layer.self_attn.head_dim

        q = decoder_layer.self_attn.q_proj(hidden_states).view(-1, num_heads, head_dim)
        k = decoder_layer.self_attn.k_proj(hidden_states).view(-1, num_kv_heads, head_dim)
        v = decoder_layer.self_attn.v_proj(hidden_states).view(-1, num_kv_heads, head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)

        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            softmax_scale=decoder_layer.self_attn.scaling,
            causal=True,
        )

        attn_output = attn_output.view(1, hidden_states.shape[1], -1)
        hidden_states = residual + decoder_layer.self_attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        cross_attention_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward pass through the cross-attention-augmented Llama decoder.

        At each backbone layer, optionally applies cross-attention (if the layer
        index is in ``cross_attention_layers``) followed by causal self-attention.

        Args:
            input_ids: Input token ids ``[1, total_len]``.
            position_ids: Position ids ``[1, total_len]``. Required for
                continuous batching to ensure proper sequence isolation.
            cu_seqlens_q: Cumulative query lengths ``[batch + 1]``.
            max_seqlen_q: Maximum query length.
            cross_attention_states: Per-layer ``(K, V)`` from the chunk encoder.
            cu_seqlens_k: Cumulative key/value lengths ``[batch + 1]``.
            max_seqlen_k: Maximum key/value length.
            past_key_values: Optional KV cache for autoregressive inference.
            inputs_embeds: Pre-computed embeddings (mutually exclusive with ``input_ids``).
            use_cache: Whether to return updated KV cache.
            output_attentions: Unused; kept for API compatibility.
            output_hidden_states: Whether to return all intermediate hidden states.
            cache_position: Token positions for cache updates.

        Returns:
            :class:`BaseModelOutputWithPast` with last hidden state, optional
            past key values, and optional all hidden states.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if position_ids is None:
            raise ValueError("position_ids must be provided for continuous batching.")

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # --- Cross-Attention Phase ---
            cross_layer_id = self._cross_layer_map.get(idx)
            have_cross_cache = (
                past_key_values is not None
                and past_key_values.get_seq_length(idx + len(self.layers)) > 0
            )

            if cross_layer_id is not None and (cross_attention_states is not None or have_cross_cache):
                layer_outputs = self.cross_layers[cross_layer_id](
                    hidden_states=hidden_states,
                    cross_attention_states=(
                        cross_attention_states[cross_layer_id] if cross_attention_states else None
                    ),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]

            # --- Self-Attention Phase ---
            if self.gradient_checkpointing and self.training:
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
                    decoder_layer, hidden_states, position_embeddings,
                    cu_seqlens_q, max_seqlen_q,
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


# ---------------------------------------------------------------------------
# LM Heads
# ---------------------------------------------------------------------------

class CombLlamaForCausalLM(CombLlamaPreTrainedModel, GenerationMixin):
    """Causal language model head on top of the augmented Llama decoder.

    Wraps :class:`CombLlamaTextModel` with a linear LM head for next-token
    prediction. Supports both standard and cross-attention-augmented generation.
    """

    config_class = LlamaConfig
    _supports_static_cache = True
    base_model_prefix = "language_model"

    def __init__(self, config):
        text_config = config.get_text_config()
        super().__init__(text_config)
        self.text_config = text_config
        self.vocab_size = text_config.vocab_size
        self.model = CombLlamaTextModel._from_config(config)
        self.lm_head = nn.Linear(text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
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
        cross_attention_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids: Input token ids ``[1, total_len]``.
            position_ids: Position ids ``[1, total_len]``.
            past_key_values: KV cache from previous generation steps.
            inputs_embeds: Pre-computed embeddings (mutually exclusive with ``input_ids``).
            labels: Labels for language modeling loss ``[1, total_len]``.
                Tokens set to ``-100`` are ignored.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            cache_position: Token positions for cache updates.
            logits_to_keep (``int`` or ``torch.Tensor``):
                If ``int > 0``, compute logits only for the last N tokens.
                If ``0``, compute for all tokens. If ``Tensor``, index directly.
            shift_labels: Pre-shifted labels ``[1, total_len]`` where
                position *n* predicts token *n* (no internal shifting needed).
            cu_seqlens_q: Cumulative query lengths ``[batch + 1]``.
            max_seqlen_q: Maximum query length in the batch.
            cross_attention_states: Per-layer ``(K, V)`` from chunk encoder.
            cu_seqlens_k: Cumulative key/value lengths ``[batch + 1]``.
            max_seqlen_k: Maximum key/value length in the batch.

        Returns:
            :class:`CausalLMOutputWithPast` with loss, logits, cache, and hidden states.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cross_attention_states=cross_attention_states,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
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


class CombLlamaForConditionalGeneration(CombLlamaPreTrainedModel, GenerationMixin):
    """Full CombLlama model: chunk encoder + cross-attention-augmented Llama decoder.

    Compresses dialogue history into chunk representations via the chunk encoder,
    then injects them into the Llama decoder through cross-attention layers.
    This reduces the KV cache memory footprint for long-context inference.

    When ``from_scratch=True``, loads pretrained Llama-3.1-8B-Instruct weights
    for the backbone, copies embedding weights to the chunk encoder, freezes
    all backbone parameters, and only trains the cross-attention layers and
    chunk encoder (excluding its embedding).

    Args:
        config (``CombLlamaConfig``): Model configuration.
        from_scratch (``bool``, *optional*, defaults to ``False``):
            If ``True``, loads pretrained Llama weights and sets up the
            freeze/train parameter strategy for training from scratch.
    """

    def __init__(self, config: CombLlamaConfig, from_scratch: bool = False):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.chunk_model = CombLlamaChunkModel._from_config(config)
        self.language_model = CombLlamaForCausalLM._from_config(config)
        self.post_init()

        if from_scratch:
            self.language_model = CombLlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct", config=config,
            )
            # Copy backbone embedding weights to chunk encoder
            self.chunk_model.embed_tokens.load_state_dict(
                self.language_model.model.embed_tokens.state_dict(),
            )
            # Freeze backbone and shared embeddings
            for param in self.language_model.parameters():
                param.requires_grad = False
            for param in self.chunk_model.embed_tokens.parameters():
                param.requires_grad = False
            # Only train cross-attention layers
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

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        chunk_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_k: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
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
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        cu_seqlens_chunk: Optional[torch.Tensor] = None,
        max_seqlen_chunk: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids: Decoder input token ids ``[1, total_input_len]``.
            chunk_ids: Chunk encoder input ids ``[1, total_chunk_len]``.
                Mutually exclusive with ``cross_attention_states``.
            cross_attention_states: Pre-computed chunk encoder outputs.
                Mutually exclusive with ``chunk_ids``.
            position_ids: Decoder position ids ``[1, total_input_len]``.
            position_ids_k: Chunk encoder position ids ``[1, total_chunk_len]``.
            past_key_values: KV cache for autoregressive generation.
            inputs_embeds: Pre-computed decoder embeddings (mutually exclusive
                with ``input_ids``).
            labels: Labels for language modeling loss.
            use_cache: Whether to use/return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            cache_position: Token positions for cache updates.
            logits_to_keep: Number of tail positions to compute logits for.
            shift_labels: Pre-shifted labels ``[1, total_input_len]``.
            cu_seqlens_q: Cumulative decoder query lengths ``[batch + 1]``.
            max_seqlen_q: Maximum decoder query length.
            cu_seqlens_k: Cumulative cross-attention key lengths ``[batch + 1]``.
            max_seqlen_k: Maximum cross-attention key length.
            cu_seqlens_chunk: Cumulative chunk self-attention lengths
                ``[num_chunks + 1]``.
            max_seqlen_chunk: Maximum individual chunk length.

        Returns:
            :class:`CausalLMOutputWithPast` with loss, logits, cache, and hidden states.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if chunk_ids is not None and inputs_embeds is not None:
            raise ValueError("chunk_ids and inputs_embeds cannot both be provided")

        if chunk_ids is not None and cross_attention_states is not None:
            raise ValueError("chunk_ids and cross_attention_states cannot both be provided")

        if chunk_ids is not None:
            cross_attention_states = self.chunk_model(
                chunk_ids, cu_seqlens_chunk, max_seqlen_chunk, position_ids_k,
            )

        outputs = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            cross_attention_states=cross_attention_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        loss = _compute_loss(outputs.logits, labels, shift_labels, self.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = CombLlamaForConditionalGeneration(
        from_scratch=True,
        config=CombLlamaConfig(LlamaConfig.from_pretrained(model_name)),
    )
