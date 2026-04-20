import pytest
import torch
from transformers import LlamaConfig

from baselines.YOCO.models.YOCO import YOCOConfig, YOCOForCausalLM


def build_tiny_model():
    text_cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=32,
        num_attention_heads=8,
        num_key_value_heads=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=8,
        max_position_embeddings=128,
    )
    config = YOCOConfig(text_config=text_cfg, num_self_decoder_layers=16, num_cross_decoder_layers=16)
    return YOCOForCausalLM(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for flash-attn smoke tests")
def test_yoco_smoke_forward_and_generate():
    model = build_tiny_model().eval().to("cuda", dtype=torch.bfloat16)
    input_ids = torch.tensor([[1, 5, 6, 7]], device="cuda")
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
    shift_labels = input_ids.clone()
    cu_seqlens_q = torch.tensor([0, input_ids.shape[1]], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=input_ids.shape[1],
            use_cache=False,
        )

    assert outputs.logits.shape == (1, input_ids.shape[1], model.vocab_size)
    assert torch.isfinite(outputs.loss)

    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False, use_cache=True)

    assert generated.shape[1] == input_ids.shape[1] + 2
