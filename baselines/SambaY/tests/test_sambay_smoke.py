import torch
from transformers import LlamaConfig

from baselines.SambaY.models.SambaY import SambaYConfig, SambaYForCausalLM


def build_tiny_model():
    text_cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=128,
    )
    config = SambaYConfig(
        text_config=text_cfg,
        num_self_decoder_layers=4,
        num_cross_decoder_layers=4,
        sliding_window=4,
    )
    return SambaYForCausalLM(config)


def test_sambay_smoke_forward_cpu():
    model = build_tiny_model().eval()
    input_ids = torch.tensor([[1, 5, 6, 7]])
    shift_labels = torch.tensor([[-100, 5, 6, 7]])
    cu_seqlens_q = torch.tensor([0, input_ids.shape[1]], dtype=torch.int32)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            shift_labels=shift_labels,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=input_ids.shape[1],
            use_cache=False,
        )

    assert outputs.logits.shape == (1, input_ids.shape[1], model.vocab_size)
    assert torch.isfinite(outputs.loss)


def test_sambay_cross_decoder_contains_gmu_and_cross_attention():
    model = build_tiny_model()
    layer_types = [layer.use_gmu for layer in model.model.cross_decoder.layers]

    assert any(layer_types)
    assert any(not use_gmu for use_gmu in layer_types)


def test_sambay_self_decoder_has_boundary_full_attention_block():
    model = build_tiny_model()

    assert len(model.model.self_decoder.layers) == model.config.num_self_decoder_layers
    assert model.model.self_decoder.gmu_save_layer.use_mamba
    assert hasattr(model.model.self_decoder, "boundary_layer")
    assert hasattr(model.model.self_decoder.boundary_layer, "self_attn")
