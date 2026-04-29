"""SambaY-native training datasets."""

from baselines.SambaY.data.chat_datasets import (
    ChatsV2SambaYDataset,
    ClaudeSambaYDataset,
    InfinityInstructSambaYDataset,
    MagpieUltraSambaYDataset,
    UltraChatSambaYDataset,
    WildChatSambaYDataset,
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
    "ultrachat_200k": UltraChatSambaYDataset,
    "chats_v2_k200000": ChatsV2SambaYDataset,
    "allenai-WildChat-4.8M": WildChatSambaYDataset,
    "claude": ClaudeSambaYDataset,
    "infinity-instruct": InfinityInstructSambaYDataset,
    "magpie-ultra": MagpieUltraSambaYDataset,
}

