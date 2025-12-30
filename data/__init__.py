from .Ultrachat import UltrachatDataset

TRAIN_DATASETS = ["ultrachat_200k"]
TEST_DATASETS = []

DATASET_DICT = {
    "ultrachat_200k": UltrachatDataset
}