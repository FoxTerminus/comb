"""Comb: chunk encoder + cross-attention augmented Qwen3 decoder.

Adapted from ``archive/CombLlama.py``.  Drops all Llama dependencies;
uses Qwen3 components exclusively.

cross_attention_layers = [3,7,11,15,19,23,27]  (7 layers, 1/4 of 28)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from .config import CombConfig

# ---------------------------------------------------------------------------
# Chunked loss (avoids materialising full [B, T, vocab] logits)
# ---------------------------------------------------------------------------

def _chunked_loss(lm_head, hidden_states, shift_labels, chunk_size=512):
    """Compute cross-entropy loss in token chunks to save GPU memory.

    For 32K target × 152K vocab, full fp32 logits would be ~18.6 GiB.
    Chunking avoids this peak by only materialising one chunk's logits at a time.
    """
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    B, T, D = hidden_states.shape
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        hs_chunk = hidden_states[:, start:end, :]
        labels_chunk = shift_labels[:, start:end]
        logits_chunk = lm_head(hs_chunk).float()  # [B, chunk, vocab]
        total_loss += loss_fct(
            logits_chunk.reshape(-1, logits_chunk.shape[-1]),
            labels_chunk.reshape(-1),
        )
        total_tokens += (labels_chunk != -100).sum()
    return total_loss / total_tokens.clamp(min=1)


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Cross-attention: decoder Q attends to chunk encoder K/V."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, layer_idx: int, num_decoder_layers: int,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx + num_decoder_layers  # offset cache index
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        if cross_attention_states is not None:
            k, v = cross_attention_states  # (total_kv_len, n_kv_heads, head_dim)
            k = self.k_norm(k)
            k = k.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, self.num_kv_heads, self.head_dim)
            v = v.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, self.num_kv_heads, self.head_dim)
        elif past_key_value is not None and cache_position[0] != 0:
            k, v = past_key_value[self.layer_idx]
            k = k.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
            v = v.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
        else:
            raise ValueError("cross_attention_states or past_key_value required")

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, T], dtype=torch.int32, device=hidden_states.device)
            cu_seqlens_k = cu_seqlens_q
            max_seqlen_q = T
            max_seqlen_k = T
        if hidden_states.is_cuda:
            y = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                causal=False, softmax_scale=self.scaling,
            )
        else:
            # CPU fallback: handle both 3D (varlen) and 4D (expanded) shapes
            repeats = self.num_heads // self.num_kv_heads
            if k.dim() == 4:
                if repeats > 1:
                    k = k.repeat_interleave(repeats, dim=2)
                    v = v.repeat_interleave(repeats, dim=2)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=False, scale=self.scaling,
                ).squeeze(0).transpose(0, 1)
            else:
                if repeats > 1:
                    k = k.repeat_interleave(repeats, dim=1)
                    v = v.repeat_interleave(repeats, dim=1)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                    is_causal=False, scale=self.scaling,
                ).squeeze(0).transpose(0, 1)
        y = y.reshape(B, T, -1)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Chunk Encoder
# ---------------------------------------------------------------------------

class ChunkVarlenAttention(nn.Module):
    """Bidirectional self-attention for the chunk encoder."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings,
                cu_seqlens=None, max_seqlen=None):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(2)  # (B, T, 1, head_dim)
        sin = sin.unsqueeze(2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=hidden_states.device)
            max_seqlen = T
        if hidden_states.is_cuda:
            y = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=False, softmax_scale=self.scaling,
            )
        else:
            # CPU fallback: repeat K/V for GQA, then SDPA
            repeats = self.num_heads // self.num_kv_heads
            if repeats > 1:
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=False, scale=self.scaling,
            ).squeeze(0).transpose(0, 1)
        y = y.reshape(B, T, -1)
        return self.o_proj(y)


class ChunkVarlenLayer(nn.Module):
    """One chunk encoder transformer block."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.attn_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ChunkVarlenAttention(
            hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
        )
        self.mlp_norm = Qwen3RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states, position_embeddings,
                cu_seqlens=None, max_seqlen=None):
        residual = hidden_states
        hidden_states = residual + self.self_attn(
            self.attn_norm(hidden_states), position_embeddings,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        residual = hidden_states
        hidden_states = residual + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class CombChunkModel(nn.Module):
    """Bidirectional chunk encoder with per-layer K/V projections."""

    def __init__(self, config: CombConfig):
        super().__init__()
        cfg = config.text_config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)

        self.layers = nn.ModuleList([
            ChunkVarlenLayer(cfg) for _ in range(config.num_cross_layers)
        ])

        self.k_proj = nn.ModuleList([
            nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
            for _ in range(config.num_cross_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
            for _ in range(config.num_cross_layers)
        ])

        self.rotary_emb = Qwen3RotaryEmbedding(cfg)

    def forward(self, chunk_ids, cu_seqlens_chunk=None, max_seqlen_chunk=None,
                position_ids_k=None):
        hidden_states = self.embed_tokens(chunk_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids_k)
        cross_attention_states = []

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, position_embeddings,
                cu_seqlens=cu_seqlens_chunk, max_seqlen=max_seqlen_chunk,
            )
            k = self.k_proj[idx](hidden_states).view(-1, self.num_kv_heads, self.head_dim)
            v = self.v_proj[idx](hidden_states).view(-1, self.num_kv_heads, self.head_dim)
            cross_attention_states.append((k, v))

        return cross_attention_states

    @property
    def num_kv_heads(self):
        return self.layers[0].self_attn.num_kv_heads

    @property
    def head_dim(self):
        return self.layers[0].self_attn.head_dim


# ---------------------------------------------------------------------------
# Cross-Attention Decoder Layer
# ---------------------------------------------------------------------------

class CombCrossAttentionDecoderLayer(nn.Module):
    """Decoder layer with cross-attention inserted BEFORE self-attention.
    Uses zero-initialized tanh gates to preserve backbone behavior at init."""

    def __init__(self, config: CombConfig, layer_idx: int):
        super().__init__()
        cfg = config.text_config
        self.attn_norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.cross_attn = CrossAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            layer_idx=layer_idx,
            num_decoder_layers=cfg.num_hidden_layers,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        self.mlp_norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, cross_attention_states=None,
                cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                max_seqlen_k=None, past_key_value=None, cache_position=None):
        residual = hidden_states
        ca_out = self.cross_attn(
            self.attn_norm(hidden_states),
            cross_attention_states=cross_attention_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            past_key_value=past_key_value, cache_position=cache_position,
        )
        hidden_states = residual + ca_out * torch.tanh(self.gate_attn)

        residual = hidden_states
        hidden_states = residual + self.mlp(self.mlp_norm(hidden_states)) * torch.tanh(self.gate_mlp)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder (Text Model)
# ---------------------------------------------------------------------------

class CombTextModel(PreTrainedModel):
    """Qwen3 decoder augmented with interleaved cross-attention layers."""

    config_class = CombConfig
    base_model_prefix = "model"

    def __init__(self, config: CombConfig):
        super().__init__(config)
        cfg = config.text_config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)

        self.decoder_layers = nn.ModuleList([
            Qwen3DecoderLayer(cfg, layer_idx=i)
            for i in range(cfg.num_hidden_layers)
        ])

        self.cross_layers = nn.ModuleList([
            CombCrossAttentionDecoderLayer(config, layer_idx=i)
            for i, _ in enumerate(config.cross_attention_layers)
        ])

        self._cross_layer_map = {
            blk_idx: ca_idx
            for ca_idx, blk_idx in enumerate(config.cross_attention_layers)
        }

        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(cfg)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                cross_attention_states=None, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=None, max_seqlen_k=None, past_key_values=None,
                inputs_embeds=None, use_cache=False, output_hidden_states=False,
                cache_position=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.decoder_layers):
            if idx in self._cross_layer_map:
                ca_idx = self._cross_layer_map[idx]
                ca_state = cross_attention_states[ca_idx] if cross_attention_states else None
                hidden_states = self.cross_layers[ca_idx](
                    hidden_states,
                    cross_attention_states=ca_state,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                    past_key_value=past_key_values, cache_position=cache_position,
                )

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


# ---------------------------------------------------------------------------
# Causal LM + Conditional Generation
# ---------------------------------------------------------------------------

class CombForCausalLM(PreTrainedModel, GenerationMixin):
    """Comb causal LM wrapper."""

    config_class = CombConfig
    base_model_prefix = "language_model"

    def __init__(self, config: CombConfig):
        super().__init__(config)
        self.model = CombTextModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, labels=None, shift_labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, **kwargs)
        loss = None
        logits = None
        if shift_labels is not None:
            loss = _chunked_loss(self.lm_head, outputs.last_hidden_state, shift_labels)
        elif labels is not None:
            loss = _chunked_loss(self.lm_head, outputs.last_hidden_state, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


class CombForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Comb: chunk encoder + cross-attention augmented Qwen3 decoder."""

    config_class = CombConfig
    base_model_prefix = "comb"

    def __init__(self, config: CombConfig, from_scratch: bool = True):
        super().__init__(config)
        self.chunk_model = CombChunkModel(config)
        self.language_model = CombForCausalLM(config)

        if from_scratch:
            self._init_from_scratch()

        self._fix_causal_mask()
        self.post_init()

    def _fix_causal_mask(self):
        """Ensure Qwen3 decoder self-attention uses causal SDPA.

        CombConfig's super().__init__ resets _attn_implementation to None,
        so Qwen3DecoderLayer skips causal masking. Patch all layers.
        """
        for layer in self.language_model.model.decoder_layers:
            layer.self_attn.config._attn_implementation = "sdpa"

    def _init_from_scratch(self):
        """Initialize Comb from a pretrained Qwen3-0.6B backbone."""
        from transformers import Qwen3ForCausalLM

        backbone = Qwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
        )
        bb = backbone.model  # Qwen3Model
        bb_cfg = self.config.text_config
        n_cross = self.config.num_cross_layers

        # --- 1. Copy backbone decoder layers ---
        for i, blk in enumerate(self.language_model.model.decoder_layers):
            blk.load_state_dict(bb.layers[i].state_dict())

        # --- 2. Copy embedding ---
        self.language_model.model.embed_tokens.load_state_dict(
            bb.embed_tokens.state_dict()
        )
        self.chunk_model.embed_tokens.load_state_dict(
            bb.embed_tokens.state_dict()
        )

        # --- 3. Copy LM head ---
        self.language_model.lm_head.load_state_dict(
            backbone.lm_head.state_dict()
        )

        # --- 4. Copy final norm ---
        self.language_model.model.norm.load_state_dict(
            bb.norm.state_dict()
        )

        # --- 5. Copy rotary embedding ---
        self.language_model.model.rotary_emb.load_state_dict(
            bb.rotary_emb.state_dict()
        )
        self.chunk_model.rotary_emb.load_state_dict(
            bb.rotary_emb.state_dict()
        )

        # --- 6. Copy encoder layers from decoder backbone ---
        for i, enc_layer in enumerate(self.chunk_model.layers):
            bb_idx = self.config.cross_attention_layers[i]
            # Copy attention
            enc_layer.self_attn.q_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.q_proj.state_dict()
            )
            enc_layer.self_attn.k_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.k_proj.state_dict()
            )
            enc_layer.self_attn.v_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.v_proj.state_dict()
            )
            enc_layer.self_attn.o_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.o_proj.state_dict()
            )
            enc_layer.mlp.load_state_dict(
                bb.layers[bb_idx].mlp.state_dict()
            )
            enc_layer.attn_norm.load_state_dict(
                bb.layers[bb_idx].input_layernorm.state_dict()
            )
            enc_layer.mlp_norm.load_state_dict(
                bb.layers[bb_idx].post_attention_layernorm.state_dict()
            )

        # --- 7. Encoder K/V proj from decoder K/V proj ---
        for i, (k_proj, v_proj) in enumerate(zip(
            self.chunk_model.k_proj, self.chunk_model.v_proj
        )):
            bb_idx = self.config.cross_attention_layers[i]
            k_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.k_proj.state_dict()
            )
            v_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.v_proj.state_dict()
            )

        # --- 8. Cross-attention Q/O from decoder Q/O ---
        for i, ca_layer in enumerate(self.language_model.model.cross_layers):
            bb_idx = self.config.cross_attention_layers[i]
            ca_layer.cross_attn.q_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.q_proj.state_dict()
            )
            ca_layer.cross_attn.o_proj.load_state_dict(
                bb.layers[bb_idx].self_attn.o_proj.state_dict()
            )
            ca_layer.mlp.load_state_dict(
                bb.layers[bb_idx].mlp.state_dict()
            )
            ca_layer.attn_norm.load_state_dict(
                bb.layers[bb_idx].input_layernorm.state_dict()
            )
            ca_layer.mlp_norm.load_state_dict(
                bb.layers[bb_idx].post_attention_layernorm.state_dict()
            )

        # --- 9. Freeze backbone ---
        for p in self.language_model.model.decoder_layers.parameters():
            p.requires_grad = False
        for p in self.language_model.model.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.language_model.model.norm.parameters():
            p.requires_grad = False
        for p in self.language_model.lm_head.parameters():
            p.requires_grad = False
        for p in self.language_model.model.rotary_emb.parameters():
            p.requires_grad = False
        for p in self.chunk_model.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.chunk_model.rotary_emb.parameters():
            p.requires_grad = False

        # cross-attention layers: gates start at zero
        for ca_layer in self.language_model.model.cross_layers:
            ca_layer.gate_attn.data.zero_()
            ca_layer.gate_mlp.data.zero_()

        del backbone
        torch.cuda.empty_cache()

    def forward(self, input_ids=None, chunk_ids=None, labels=None, shift_labels=None,
                cu_seqlens_q=None, cu_seqlens_k=None, cu_seqlens_chunk=None,
                max_seqlen_q=None, max_seqlen_k=None, max_seqlen_chunk=None,
                position_ids=None, position_ids_k=None, **kwargs):
        cross_attention_states = None
        if chunk_ids is not None:
            cross_attention_states = self.chunk_model(
                chunk_ids,
                cu_seqlens_chunk=cu_seqlens_chunk,
                max_seqlen_chunk=max_seqlen_chunk,
                position_ids_k=position_ids_k,
            )

        outputs = self.language_model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            labels=labels, shift_labels=shift_labels,
            position_ids=position_ids, **kwargs,
        )
        return outputs
