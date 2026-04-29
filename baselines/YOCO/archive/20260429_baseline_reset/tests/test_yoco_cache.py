import pytest
import torch
from transformers import LlamaConfig

from baselines.YOCO.models.YOCO import YOCOConfig, YOCOForCausalLM, YOCODynamicCache


def build_tiny_model():
    text_cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=32,
        num_attention_heads=8,
        num_key_value_heads=8,
        pad_token_id=0,
        sliding_window=8,
        max_position_embeddings=128,
    )
    config = YOCOConfig(text_config=text_cfg, num_self_decoder_layers=16, num_cross_decoder_layers=16)
    return YOCOForCausalLM(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cache tests")
def test_yoco_cache_matches_full_forward():
    torch.manual_seed(0)
    model = build_tiny_model().eval().to("cuda", dtype=torch.bfloat16)

    input_ids = torch.randint(0, model.vocab_size, (1, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0)
    cu_full = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    cu_prefill = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    cu_decode = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        full = model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_full,
            max_seqlen_q=8,
            use_cache=True,
        )
        assert isinstance(full.past_key_values, YOCODynamicCache)

        prefill = model(
            input_ids=input_ids[:, :7],
            position_ids=position_ids[:, :7],
            cu_seqlens_q=cu_prefill,
            max_seqlen_q=7,
            use_cache=True,
        )
        decode = model(
            input_ids=input_ids[:, 7:8],
            position_ids=position_ids[:, 7:8],
            cu_seqlens_q=cu_decode,
            max_seqlen_q=1,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )

    full_last = full.logits[:, -1, :].float()
    decode_last = decode.logits[:, -1, :].float()
    assert torch.allclose(full_last, decode_last, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cache tests")
def test_yoco_prefill_with_cache_matches_no_cache_logits():
    torch.manual_seed(0)
    model = build_tiny_model().eval().to("cuda", dtype=torch.bfloat16)

    input_ids = torch.randint(0, model.vocab_size, (1, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0)
    shift_labels = torch.roll(input_ids, shifts=-1, dims=1)
    shift_labels[:, -1] = -100
    cu_full = torch.tensor([0, 8], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        no_cache = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            cu_seqlens_q=cu_full,
            max_seqlen_q=8,
            use_cache=False,
        )
        with_cache = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            cu_seqlens_q=cu_full,
            max_seqlen_q=8,
            use_cache=True,
        )

    assert torch.allclose(no_cache.logits.float(), with_cache.logits.float(), atol=1e-2, rtol=1e-2)
    assert torch.isfinite(with_cache.loss)
