"""This file processes the Ultrachat dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class UltrachatDataset(DatasetBase):
    name = "ultrachat_200k"

    def _init_data(self, split):
        self.data = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

if __name__ == "__main__":
    model_name = "XiaomiMiMo/MiMo-V2-Flash"
    dataset = UltrachatDataset(model_name, split="train_sft")
    print(dataset[0])