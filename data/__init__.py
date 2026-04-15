from .Ultrachat import UltrachatDataset
from .ChatsV2 import ChatsV2Dataset
from .WildChat import WildChatDataset
from .claude import ClaudeDataset
from .InfinityInstruct import InfinityInstructDataset
from .MagpieUltra import MagpieUltraDataset

TRAIN_DATASETS = ["ultrachat_200k", "chats_v2_k200000", "allenai-WildChat-4.8M", "claude", "infinity-instruct", "magpie-ultra"]
TEST_DATASETS = []

DATASET_DICT = {
    "ultrachat_200k": UltrachatDataset,
    "chats_v2_k200000": ChatsV2Dataset,
    "allenai-WildChat-4.8M": WildChatDataset,
    "claude": ClaudeDataset,
    "infinity-instruct": InfinityInstructDataset,
    "magpie-ultra": MagpieUltraDataset,
}