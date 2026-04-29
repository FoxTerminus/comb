from baselines.YOCO.data.base import next_token_assistant_labels


def test_next_token_assistant_labels_masks_non_assistant_targets():
    input_ids = [10, 11, 12, 13, 14]
    same_position_labels = [-100, -100, 12, 13, 14]

    labels = next_token_assistant_labels(input_ids, same_position_labels)

    assert labels == [-100, 12, 13, 14, -100]
