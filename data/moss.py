"""This file processes the moss dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class MossDataset(DatasetBase):
    name = "moss"

    def _init_data(self, split):
        self.data = load_dataset("OpenMOSS-Team/moss-002-sft-data", split=split)
        self.data = self.data.map(self._prepare_data, remove_columns=['plain_text'], num_proc=CPU_NUM)
        
    def _prepare_data(self, example):
        plain_text = example.get('plain_text', '')
        turns = plain_text.replace('<eoh>', '<eoh>|').replace('<eoa>', '<eoa>|').split('|')
        messages = []
        for turn in turns:
            turn = turn.strip()
            if not turn:
                continue
            if turn.startswith('[Human]:'):
                content = turn.replace('[Human]:', '').strip()
                messages.append({
                    "role": "user",
                    "content": content
                })
            elif turn.startswith('[MOSS]:'):
                content = turn.replace('[MOSS]:', '').strip()
                messages.append({
                    "role": "assistant",
                    "content": content
                })
        example['messages'] = messages
        return example

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = MossDataset(model_name, split="train")
    print(dataset[0])