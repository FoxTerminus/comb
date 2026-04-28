from baselines.SambaY.data import DATASET_DICT, TRAIN_DATASETS
from baselines.SambaY.data.base import messages_to_sambay_sample, normalize_agentlans_messages


class FakeChatTokenizer:
    pad_token = None
    pad_token_id = None

    def apply_chat_template(self, messages, add_generation_prompt=False):
        role_prefix = {"system": 10, "user": 20, "assistant": 30}
        token_ids = []
        for idx, message in enumerate(messages):
            token_ids.extend([role_prefix[message["role"]], idx + 1, 128009])
        return token_ids


def test_messages_to_sambay_sample_supervises_all_assistant_turns():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]

    sample = messages_to_sambay_sample(messages, FakeChatTokenizer())

    assert sample["turn_count"] == 5
    assert sample["input_ids"] == [10, 1, 128009, 20, 2, 128009, 30, 3, 128009, 20, 4, 128009, 30, 5, 128009]
    assert sample["supervised_token_length"] > 0
    supervised_targets = [label for label in sample["shift_labels"] if label != -100]
    assert supervised_targets == [30, 3, 128009, 30, 5, 128009]


def test_normalize_agentlans_messages_maps_roles():
    conversations = [
        {"from": "system", "value": "s"},
        {"from": "human", "value": "u"},
        {"from": "gpt", "value": "a"},
    ]

    assert normalize_agentlans_messages(conversations) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]


def test_sambay_dataset_registry_contains_supported_sources():
    assert set(TRAIN_DATASETS) == set(DATASET_DICT)
    assert "allenai-WildChat-4.8M" in DATASET_DICT
    assert "chats_v2_k200000" in DATASET_DICT

