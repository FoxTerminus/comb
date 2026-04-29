import torch

from baselines.SambaY.tests.test_sambay_smoke import build_tiny_model


def test_sambay_cache_prefill_decode_matches_full_forward():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    input_ids = torch.tensor([[1, 5, 6, 7, 8]])
    cu_full = torch.tensor([0, input_ids.shape[1]], dtype=torch.int32)

    with torch.no_grad():
        full = model(
            input_ids=input_ids,
            cu_seqlens_q=cu_full,
            max_seqlen_q=input_ids.shape[1],
            use_cache=False,
        ).logits

        prefill_ids = input_ids[:, :-1]
        prefill = model(
            input_ids=prefill_ids,
            cu_seqlens_q=torch.tensor([0, prefill_ids.shape[1]], dtype=torch.int32),
            max_seqlen_q=prefill_ids.shape[1],
            use_cache=True,
        )
        decode = model(
            input_ids=input_ids[:, -1:],
            past_key_values=prefill.past_key_values,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            max_seqlen_q=1,
            use_cache=True,
        ).logits

    assert prefill.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(decode[:, -1, :], full[:, -1, :], atol=1e-5, rtol=1e-4)


def test_sambay_greedy_generate_uses_cache():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    input_ids = torch.tensor([[1, 5, 6, 7]])

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        )

    assert generated.shape == (1, input_ids.shape[1] + 2)

