"""This file processes the lmsys-chat dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class lmsysChatDataset(DatasetBase):
    name = "lmsys-chat"

    def _init_data(self, split):
        self.data = load_dataset("lmsys/lmsys-chat-1m", split=split)
        self.data = self.data.map(lambda x: {'messages': x['conversation']}, remove_columns=['conversation'], num_proc=CPU_NUM)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = lmsysChatDataset(model_name, split="train")
    print(dataset[1])