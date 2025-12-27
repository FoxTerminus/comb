from .Ultrachat import UltrachatDataset
from .lmsysChat import lmsysChatDataset
from .moss import MossDataset

TRAIN_DATASETS = ["ultrachat_200k", "lmsys-chat", "moss"]
TEST_DATASETS = []

DATASET_DICT = {
    "ultrachat_200k": UltrachatDataset,
    "lmsys-chat": lmsysChatDataset,
    "moss": MossDataset
}