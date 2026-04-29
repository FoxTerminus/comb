import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from baselines.YOCO.models.YOCO import YOCOForCausalLM
from baselines.YOCO.scripts.init_yoco_from_llama import build_yoco_from_llama, save_summary, validate_saved_checkpoint


def build_tiny_llama():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=32,
        num_attention_heads=8,
        num_key_value_heads=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config)


def test_yoco_init_from_tiny_llama(tmp_path):
    llama_dir = tmp_path / "tiny_llama"
    yoco_dir = tmp_path / "tiny_yoco"

    llama = build_tiny_llama()
    llama.save_pretrained(llama_dir)

    yoco, summary = build_yoco_from_llama(str(llama_dir), torch_dtype=torch.float32)
    yoco.save_pretrained(yoco_dir)
    summary.update(validate_saved_checkpoint(str(yoco_dir), torch.float32))
    save_summary(summary, str(yoco_dir))

    reloaded = YOCOForCausalLM.from_pretrained(str(yoco_dir), torch_dtype=torch.float32)
    assert isinstance(reloaded, YOCOForCausalLM)
    assert summary["missing_keys"] == []
    assert summary["unexpected_keys"] == []

    summary_path = yoco_dir / "init_summary.json"
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded_summary["total_params"] == summary["total_params"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for post-init forward smoke")
def test_yoco_init_forward_smoke(tmp_path):
    llama_dir = tmp_path / "tiny_llama"
    yoco_dir = tmp_path / "tiny_yoco"

    llama = build_tiny_llama()
    llama.save_pretrained(llama_dir)

    yoco, summary = build_yoco_from_llama(str(llama_dir), torch_dtype=torch.float32)
    yoco.save_pretrained(yoco_dir)
    summary.update(validate_saved_checkpoint(str(yoco_dir), torch.float32))
    save_summary(summary, str(yoco_dir))

    model = YOCOForCausalLM.from_pretrained(str(yoco_dir), torch_dtype=torch.float32).eval().to("cuda", dtype=torch.bfloat16)
    input_ids = torch.tensor([[1, 5, 6, 7]], device="cuda")
    position_ids = torch.arange(4, device="cuda").unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=4,
            use_cache=False,
        )

    assert outputs.logits.shape == (1, 4, model.vocab_size)
