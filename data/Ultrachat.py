"""This file processes the Ultrachat dataset."""

from datasets import Dataset
from datasets import load_dataset
from data.base import DatasetBase, CPU_NUM

class UltrachatDataset(DatasetBase):
    name = "ultrachat_200k"

    def _init_data(self, split):
        self.data = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
        self.data = self.data.map(self._prepare_data, num_proc=CPU_NUM)
        self.data = self.data.filter(lambda x: x['token_length'] >= 1024, num_proc=CPU_NUM)
        self.data = self._adjust_format()
        
    def _prepare_data(self, example):
        dialogue_tokens = self.tokenizer.apply_chat_template(example['messages'])
        
        message_tokens = []
        start = 0
        for i, token in enumerate(dialogue_tokens):
            if token == 128009: # <|eot_id|>
                message_tokens.append(dialogue_tokens[start:i+1])
                start = i + 1
        message_tokens = message_tokens[1:]  # Exclude the first message (system prompt)
        
        token_count = 0
        turn_count = 0
        history_item = []
        history = []
        turns = []
        for msg in message_tokens:
            token_count += len(msg)
            history_item.extend(msg)
            turn_count += 1
            if token_count >= 1024 and turn_count % 2 == 0:
                turns.append(turn_count // 2)
                history.append(history_item)
                history_item = []
                token_count = 0
        
        encoder_input = []
        decoder_input = []
        label = []
        for i in range(len(history)):
            encoder = history[:i+1]
            decoder = []
            label_item = []
            encoder_input.append(encoder)
            for msg in message_tokens[turns[i]*2:]:
                decoder.extend(msg)
            decoder_input.append(decoder)
            for i, msg in enumerate(message_tokens[turns[i]*2:]):
                if i % 2 == 0:
                    label_item.extend([-100])
                else:
                    label_item.extend(msg)
            label.append(label_item)

        return {
            "token_length": len(dialogue_tokens),
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label
        }
    
    def _adjust_format(self):
        ds_new = []
        for sample in self.data:
            if len(sample['encoder_input']) == 1:
                ds_new.append({
                    "messages": list(sample['messages']),
                    "token_length": sample['token_length'],
                    "encoder_input": list(sample['encoder_input'][0]),
                    "decoder_input": list(sample['decoder_input'][0]),
                    "label": list(sample['label'][0])
                })
            else:
                for i in range(len(sample['encoder_input'])):
                    ds_new.append({
                        "messages": list(sample['messages']),
                        "token_length": sample['token_length'],
                        "encoder_input": list(sample['encoder_input'][i]),
                        "decoder_input": list(sample['decoder_input'][i]),
                        "label": list(sample['label'][i])
                    })
        return Dataset.from_list(ds_new)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = UltrachatDataset(model_name, split="train_sft")
    # print(dataset[0])
    print(f"Total samples: {len(dataset)}")