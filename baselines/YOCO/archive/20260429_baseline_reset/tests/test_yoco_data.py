import torch

from baselines.YOCO.training.data import collate_fn_yoco


def test_collate_fn_yoco_keeps_original_behavior_without_chunks():
    batch = [
        {
            "input_ids": [1, 2, 3],
            "shift_labels": [-100, 2, 3],
        }
    ]

    packed = collate_fn_yoco(batch)

    assert torch.equal(packed["input_ids"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(packed["shift_labels"], torch.tensor([[-100, 2, 3]]))
    assert torch.equal(packed["position_ids"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(packed["cu_seqlens_q"], torch.tensor([0, 3], dtype=torch.int32))
    assert packed["max_seqlen_q"] == 3


def test_collate_fn_yoco_can_convert_same_position_labels_to_next_token_labels():
    batch = [
        {
            "input_ids": [1, 2, 3],
            "shift_labels": [-100, 2, 3],
        }
    ]

    packed = collate_fn_yoco(batch, label_shift_mode="next-token")

    assert torch.equal(packed["shift_labels"], torch.tensor([[2, 3, -100]]))
