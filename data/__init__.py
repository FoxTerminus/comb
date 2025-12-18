from .Ultrachat import UltrachatDataset
from .lmsysChat import lmsysChatDataset
from .moss import MossDataset

TRAIN_DATASETS = ["ultrachat_200k", "lmsys-chat", "moss"]
TEST_DATASETS = []

DATASET_DICT = {
    "moss": MossDataset,
    "lmsys-chat": lmsysChatDataset,
    "ultrachat_200k": UltrachatDataset
}