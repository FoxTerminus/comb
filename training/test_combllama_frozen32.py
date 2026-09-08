import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from comb.integration.hf.CombLlamaFrozen32 import (
    CombLlamaFrozen32Config,
    CombLlamaFrozen32ForConditionalGeneration,
    Frozen32PICContextLayer,
)


def tiny_config() -> CombLlamaFrozen32Config:
    llama = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    return CombLlamaFrozen32Config(
        llama,
        encoder_num_hidden_layers=4,
        decoder_num_hidden_layers=4,
        cross_attention_layers=[1, 3],
        encoder_attn_implementation="eager",
        decoder_attn_implementation="eager",
        context_gate_init=1e-3,
    )


def test_config_serialization_round_trip():
    config = tiny_config()
    restored = CombLlamaFrozen32Config.from_dict(config.to_dict())
    assert restored.encoder_num_hidden_layers == 4
    assert restored.decoder_num_hidden_layers == 4
    assert restored.cross_attention_layers == [1, 3]
    assert restored.num_hidden_layers == 6
    assert restored.text_config.num_hidden_layers == 6


def test_pretrained_initialization_and_trainable_boundary():
    torch.manual_seed(7)
    config = tiny_config()
    backbone_config = LlamaConfig(**config.text_config.to_dict())
    backbone_config.num_hidden_layers = config.decoder_num_hidden_layers
    backbone_config._attn_implementation = "eager"
    backbone = LlamaForCausalLM(backbone_config)
    model = CombLlamaFrozen32ForConditionalGeneration(config)
    model.initialize_from_llama(backbone)

    assert torch.equal(
        model.chunk_model.model.layers[2].self_attn.q_proj.weight,
        backbone.model.layers[2].self_attn.q_proj.weight,
    )
    assert torch.equal(
        model.language_model.model.layers[2].mlp.down_proj.weight,
        backbone.model.layers[2].mlp.down_proj.weight,
    )

    for decoder_index, context in zip(
        config.cross_attention_layers, model.language_model.model.cross_layers
    ):
        source = backbone.model.layers[decoder_index]
        assert torch.equal(context.cross_attn.q_proj.weight, source.self_attn.q_proj.weight)
        assert torch.equal(context.cross_attn.k_proj.weight, source.self_attn.k_proj.weight)
        assert torch.equal(context.cross_attn.v_proj.weight, source.self_attn.v_proj.weight)
        assert torch.equal(context.cross_attn.o_proj.weight, source.self_attn.o_proj.weight)
        assert torch.equal(context.mlp.gate_proj.weight, source.mlp.gate_proj.weight)
        assert torch.equal(context.mlp.up_proj.weight, source.mlp.up_proj.weight)
        assert torch.equal(context.mlp.down_proj.weight, source.mlp.down_proj.weight)
        assert context.context_gate.item() == torch.tensor(1e-3).item()

    trainable = model.trainable_parameter_names()
    assert trainable
    assert all("language_model.model.cross_layers" in name for name in trainable)
    assert not any(parameter.requires_grad for parameter in model.chunk_model.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.model.layers.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.lm_head.parameters())


class _ZeroContext(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states)


def test_context_mlp_has_no_query_only_path():
    llama = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    layer = Frozen32PICContextLayer(llama, layer_idx=1, gate_init=0.5)
    layer.cross_attn = _ZeroContext()
    hidden = torch.randn(2, 3, llama.hidden_size)
    output = layer(
        hidden,
        cross_attention_states=torch.randn(2, 5, llama.hidden_size),
        cross_attention_mask=None,
    )
    assert torch.equal(output, hidden)


def test_pic_shape_forward_and_first_step_gradients():
    torch.manual_seed(11)
    config = tiny_config()
    backbone_config = LlamaConfig(**config.text_config.to_dict())
    backbone_config.num_hidden_layers = config.decoder_num_hidden_layers
    backbone_config._attn_implementation = "eager"
    backbone = LlamaForCausalLM(backbone_config)
    model = CombLlamaFrozen32ForConditionalGeneration(config)
    model.initialize_from_llama(backbone)
    model.train()

    chunk_ids = torch.randint(1, config.text_config.vocab_size, (2, 6))
    input_ids = torch.randint(1, config.text_config.vocab_size, (2, 5))
    mask = torch.ones_like(chunk_ids)
    pic = model.build_pic_cache(chunk_ids, mask)
    assert len(pic) == 2
    assert pic[0][0].shape == (2, 6, config.text_config.num_key_value_heads, 8)
    assert pic[0][1].shape == pic[0][0].shape

    shift_labels = input_ids.clone()
    shift_labels[:, :2] = -100
    output = model(
        input_ids=input_ids,
        chunk_ids=chunk_ids,
        cross_attention_mask=mask,
        shift_labels=shift_labels,
        use_cache=False,
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()

    first = model.language_model.model.cross_layers[0]
    assert first.context_gate.grad is not None
    assert first.cross_attn.q_proj.weight.grad is not None
    assert first.cross_attn.k_proj.weight.grad is not None
    assert first.cross_attn.v_proj.weight.grad is not None
    assert first.cross_attn.o_proj.weight.grad is not None
    assert first.mlp.down_proj.weight.grad is not None
    assert torch.count_nonzero(first.cross_attn.q_proj.weight.grad) > 0
    assert torch.count_nonzero(first.mlp.down_proj.weight.grad) > 0


def initialized_pair():
    config = tiny_config()
    backbone_config = LlamaConfig(**config.text_config.to_dict())
    backbone_config.num_hidden_layers = config.decoder_num_hidden_layers
    backbone_config._attn_implementation = "eager"
    backbone = LlamaForCausalLM(backbone_config)
    model = CombLlamaFrozen32ForConditionalGeneration(config)
    model.initialize_from_llama(backbone)
    return backbone.eval(), model.eval()


def test_no_context_is_exact_backbone():
    torch.manual_seed(17)
    backbone, model = initialized_pair()
    input_ids = torch.randint(1, model.config.text_config.vocab_size, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        actual = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
    assert torch.equal(actual, expected)


def test_raw_context_and_prebuilt_pic_are_exact():
    torch.manual_seed(19)
    _, model = initialized_pair()
    input_ids = torch.randint(1, model.config.text_config.vocab_size, (2, 5))
    chunk_ids = torch.randint(1, model.config.text_config.vocab_size, (2, 9))
    chunk_mask = torch.ones_like(chunk_ids)
    with torch.no_grad():
        direct = model(
            input_ids=input_ids,
            chunk_ids=chunk_ids,
            cross_attention_mask=chunk_mask,
            use_cache=False,
        ).logits
        pic = model.build_pic_cache(chunk_ids, chunk_mask)
        cached = model(
            input_ids=input_ids,
            cross_attention_states=pic,
            cross_attention_mask=chunk_mask,
            use_cache=False,
        ).logits
    assert torch.equal(direct, cached)


def test_optimizer_step_changes_only_context_branch():
    torch.manual_seed(23)
    backbone, model = initialized_pair()
    model.train()
    frozen_before = {
        "encoder": model.chunk_model.model.layers[0].self_attn.q_proj.weight.detach().clone(),
        "decoder": model.language_model.model.layers[0].self_attn.q_proj.weight.detach().clone(),
        "head": model.language_model.lm_head.weight.detach().clone(),
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    input_ids = torch.randint(1, model.config.text_config.vocab_size, (2, 5))
    chunk_ids = torch.randint(1, model.config.text_config.vocab_size, (2, 8))
    labels = input_ids.clone()
    labels[:, :2] = -100
    loss = model(
        input_ids=input_ids,
        chunk_ids=chunk_ids,
        cross_attention_mask=torch.ones_like(chunk_ids),
        shift_labels=labels,
        use_cache=False,
    ).loss
    loss.backward()
    optimizer.step()

    assert torch.equal(
        frozen_before["encoder"],
        model.chunk_model.model.layers[0].self_attn.q_proj.weight,
    )
    assert torch.equal(
        frozen_before["decoder"],
        model.language_model.model.layers[0].self_attn.q_proj.weight,
    )
    assert torch.equal(frozen_before["head"], model.language_model.lm_head.weight)
    assert torch.equal(
        model.language_model.model.layers[0].self_attn.q_proj.weight,
        backbone.model.layers[0].self_attn.q_proj.weight,
    )
