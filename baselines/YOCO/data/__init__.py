"""YOCO-native training datasets."""

from baselines.YOCO.data.base import CPU_NUM
from baselines.YOCO.data.chat_datasets import (
    ChatsV2YOCODataset,
    ClaudeYOCODataset,
    InfinityInstructYOCODataset,
    MagpieUltraYOCODataset,
    UltraChatYOCODataset,
    WildChatYOCODataset,
)

TRAIN_DATASETS = [
    "ultrachat_200k",
    "chats_v2_k200000",
    "allenai-WildChat-4.8M",
    "claude",
    "infinity-instruct",
    "magpie-ultra",
]

DATASET_DICT = {
    "ultrachat_200k": UltraChatYOCODataset,
    "chats_v2_k200000": ChatsV2YOCODataset,
    "allenai-WildChat-4.8M": WildChatYOCODataset,
    "claude": ClaudeYOCODataset,
    "infinity-instruct": InfinityInstructYOCODataset,
    "magpie-ultra": MagpieUltraYOCODataset,
}
