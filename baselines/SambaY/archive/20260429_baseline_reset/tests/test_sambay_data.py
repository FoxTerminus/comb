import torch

from baselines.SambaY.training.data import collate_fn_sambay, preprocess_sambay_example


def test_preprocess_sambay_example_drops_comb_fields():
    item = {
        "input_ids": [1, 5, 6],
        "shift_labels": [-100, 5, 6],
        "chunk_ids": [99, 100],
        "cross_attention_states": "ignored",
    }

    processed = preprocess_sambay_example(item)

    assert set(processed) == {"input_ids", "shift_labels", "attention_mask", "sequence_id", "seq_len"}
    assert torch.equal(processed["input_ids"], torch.tensor([1, 5, 6]))
    assert "chunk_ids" not in processed


def test_collate_fn_sambay_packs_sequence_boundaries():
    batch = [
        {"input_ids": [1, 2, 3], "shift_labels": [-100, 2, 3], "chunk_ids": [7]},
        {"input_ids": [4, 5], "shift_labels": [-100, 5], "chunk_ids": [8]},
    ]

    packed = collate_fn_sambay(batch, include_position_ids=True)

    assert torch.equal(packed["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(packed["shift_labels"], torch.tensor([[-100, 2, 3, -100, 5]]))
    assert torch.equal(packed["cu_seqlens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))
    assert packed["max_seqlen_q"] == 3
    assert torch.equal(packed["sequence_ids"], torch.tensor([[0, 0, 0, 1, 1]]))
    assert torch.equal(packed["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
    assert "chunk_ids" not in packed


def test_collate_fn_sambay_can_convert_same_position_labels():
    batch = [{"input_ids": [1, 2, 3], "shift_labels": [-100, 2, 3]}]

    packed = collate_fn_sambay(batch, label_shift_mode="next-token")

    assert torch.equal(packed["shift_labels"], torch.tensor([[2, 3, -100]]))

