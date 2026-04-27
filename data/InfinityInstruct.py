"""This file processes the InfinityInstruct dataset."""

from datasets import Features, Sequence, Value
from datasets import load_dataset
from data.base import DatasetBase, CPU_NUM, HF_HOME

EOT_TOKEN_ID = 128009  # <|eot_id|>
CHUNK_THRESHOLD = 1024
UPPER_TOKEN_LIMIT = 8 * 1024


class InfinityInstructDataset(DatasetBase):
    name = "infinity-instruct"

    def _init_data(self, split):
        features = Features({
            "token_length": Value("int64"),
            "chunk_num": Value("int64"),
            "chunk_ids": Sequence(Sequence(Value("int64"))),
            "input_ids": Sequence(Value("int64")),
            "shift_labels": Sequence(Value("int64")),
        })
        self.data = load_dataset("agentlans/multiturn-chat", "infinity-instruct", split=split)
        self.data = self.data.remove_columns(["source"])
        self.data = self.data.map(self._batching, num_proc=CPU_NUM)
        self.data = self.data.map(self._prepare_data, batched=True, batch_size=1,
                remove_columns=["conversations", "message_tokens"], num_proc=CPU_NUM,
                features=features)
        self.data = self.data.filter(lambda x: x["token_length"] >= CHUNK_THRESHOLD and x["token_length"] <= UPPER_TOKEN_LIMIT,
                num_proc=CPU_NUM)
        self.data = self.data.filter(lambda x: x["chunk_num"] > 0,
                num_proc=CPU_NUM)

    def _batching(self, example):
        formatted_messages = [
            {
                "role": "system" if msg["from"] == "system" else "user" if msg["from"] == "human" else "assistant",
                "content": msg["value"],
            }
            for msg in example['conversations']
        ]
        dialogue_tokens = self.tokenizer.apply_chat_template(formatted_messages)

        message_tokens = []
        start = 0
        for i, token in enumerate(dialogue_tokens):
            if token == EOT_TOKEN_ID:
                message_tokens.append(dialogue_tokens[start:i + 1])
                start = i + 1
        if formatted_messages and formatted_messages[0]["role"] == "system":
            message_tokens = message_tokens[1:]
        return {"message_tokens": message_tokens}

    @staticmethod
    def _build_shift_labels(messages):
        """Build shift labels: mask user turns with -100, keep assistant turns."""
        shift_labels = []
        for idx, msg in enumerate(messages):
            if idx % 2 == 0:
                shift_labels.extend([-100] * len(msg))
            else:
                shift_labels.extend(msg)
        return shift_labels

    @staticmethod
    def _split_history(message_tokens):
        """Split dialogue into history chunks and corresponding turn boundaries.

        Groups consecutive turn pairs (user + assistant). When accumulated token
        count reaches CHUNK_THRESHOLD, the current accumulation becomes a history
        chunk and a new accumulation begins.

        Returns:
            history: list of token lists, each representing a compressed history chunk.
            turns: list of turn-pair indices where each chunk boundary falls.
        """
        token_count = 0
        turn_count = 0
        history_item = []
        current_dialog = []
        history = []
        turns = []

        for msg in message_tokens:
            token_count += len(msg)
            current_dialog.extend(msg)
            turn_count += 1

            if turn_count % 2 == 0:
                if token_count < CHUNK_THRESHOLD:
                    history_item.extend(current_dialog)
                else:
                    if history_item:  # Don't create a chunk from empty history
                        turns.append(turn_count // 2 - 1)
                        history.append(history_item)
                    history_item = current_dialog
                    token_count = len(history_item)
                current_dialog = []

        return history, turns

    def _prepare_data(self, batch):
        all_chunk_ids = []
        all_input_ids = []
        all_shift_labels = []
        all_token_length = []
        all_chunk_num = []

        for message_tokens in batch['message_tokens']:
            token_length = sum(len(msg) for msg in message_tokens)
            history, turns = self._split_history(message_tokens)

            if not history:
                all_token_length.append(token_length)
                all_chunk_ids.append([])
                all_chunk_num.append(0)
                all_input_ids.append([tok for msg in message_tokens for tok in msg])
                all_shift_labels.append(self._build_shift_labels(message_tokens))
                continue

            for i in range(len(history)):
                all_token_length.append(token_length)
                all_chunk_ids.append(history[:i + 1])
                all_chunk_num.append(i + 1)
                current_messages = message_tokens[turns[i] * 2:]
                all_input_ids.append([tok for msg in current_messages for tok in msg])
                all_shift_labels.append(self._build_shift_labels(current_messages))

        return {
            "token_length": all_token_length,
            "chunk_num": all_chunk_num,
            "chunk_ids": all_chunk_ids,
            "input_ids": all_input_ids,
            "shift_labels": all_shift_labels,
        }


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = InfinityInstructDataset(model_name, split="train")
    print(dataset[0])
    print(f"Total samples: {len(dataset)}")
