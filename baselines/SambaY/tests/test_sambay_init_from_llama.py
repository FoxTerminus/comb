import torch
from transformers import LlamaConfig, LlamaForCausalLM

from baselines.SambaY.scripts.init_sambay_from_llama import (
    build_sambay_from_llama,
    validate_llama_config_for_sambay,
    validate_saved_checkpoint,
)


def test_init_sambay_from_tiny_llama(tmp_path):
    llama_dir = tmp_path / "tiny_llama"
    out_dir = tmp_path / "sambay"
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    llama = LlamaForCausalLM(config)
    llama.save_pretrained(llama_dir)

    sambay, summary = build_sambay_from_llama(
        str(llama_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    sambay.save_pretrained(out_dir)
    loading_info = validate_saved_checkpoint(str(out_dir), torch_dtype=torch.float32)

    assert summary["num_self_decoder_layers"] == 4
    assert summary["num_cross_decoder_layers"] == 4
    assert summary["archscale_schedule"] == {
        "self_decoder_local_layers": [0, 3],
        "gmu_save_layer": 4,
        "boundary_full_attention_layer": 5,
        "cross_decoder_layers": [6, 7],
    }
    assert summary["actual_module_counts"] == {
        "self_decoder_local_layers": 4,
        "gmu_save_layers": 1,
        "boundary_full_attention_layers": 1,
        "cross_decoder_layers": 2,
        "total_arch_layers": 8,
    }
    assert summary["cross_decoder_global_layer_indices"] == [6, 7]
    assert summary["cross_decoder_uses_gmu"] == [True, False]
    assert summary["newly_initialized_modules"] == [
        "Samba/Mamba token mixers, including the forced boundary gmu_save mixer",
        "GMU layers",
    ]
    assert "model.self_decoder.gmu_save_layer.mlp.gate_proj.weight" in summary["copied_parameter_groups"]
    assert not any(
        "gmu_save_layer.token_mixer" in parameter_group
        for parameter_group in summary["copied_parameter_groups"]
    )
    assert loading_info["missing_keys"] == []
    assert loading_info["unexpected_keys"] == []


def test_init_rejects_non_archscale_layer_count():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    try:
        validate_llama_config_for_sambay(config)
    except ValueError as exc:
        assert "divisible by 4" in str(exc)
    else:
        raise AssertionError("Expected non-ArchScale layer count to be rejected.")
