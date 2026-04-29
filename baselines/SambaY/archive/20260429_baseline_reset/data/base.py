"""SambaY-native multi-turn SFT dataset preprocessing.

This module starts from raw chat datasets and builds decoder-only SambaY
training samples. It intentionally does not consume CombLlama chunked examples.
Each full multi-turn conversation is chat-template tokenized, assistant turns
are supervised, and user/system turns are masked with ``-100`` labels.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from datasets import Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer

CPU_NUM = os.cpu_count() or 1
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
EOT_TOKEN_ID = 128009
DEFAULT_MAX_TOKEN_LENGTH = 8 * 1024

SAMBAY_FEATURES = Features(
    {
        "token_length": Value("int64"),
        "turn_count": Value("int64"),
        "supervised_token_length": Value("int64"),
        "input_ids": Sequence(Value("int64")),
        "shift_labels": Sequence(Value("int64")),
    }
)


def normalize_agentlans_messages(conversations):
    """Normalize agentlans/multiturn-chat records to HF chat-template roles."""
    return [
        {
            "role": "system" if msg["from"] == "system" else "user" if msg["from"] == "human" else "assistant",
            "content": msg["value"],
        }
        for msg in conversations
    ]


def split_chat_template_by_eot(token_ids, eot_token_id: int = EOT_TOKEN_ID):
    """Split chat-template tokens into per-message spans by EOT token."""
    message_tokens = []
    start = 0
    for idx, token in enumerate(token_ids):
        if token == eot_token_id:
            message_tokens.append(token_ids[start : idx + 1])
            start = idx + 1
    if start < len(token_ids):
        message_tokens.append(token_ids[start:])
    return message_tokens


def next_token_assistant_labels(input_ids, same_position_labels):
    """Build causal labels for positions whose next token is assistant-supervised."""
    labels = [-100] * len(input_ids)
    for idx in range(len(input_ids) - 1):
        if same_position_labels[idx + 1] != -100:
            labels[idx] = input_ids[idx + 1]
    return labels


def messages_to_sambay_sample(messages, tokenizer, max_input_length: int | None = None):
    """Tokenize one complete multi-turn chat into a SambaY decoder sample."""
    token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    message_tokens = split_chat_template_by_eot(token_ids)

    # Some Llama chat templates emit a leading header/BOS segment before the
    # first EOT. Merge it into the first message span when detected.
    if len(message_tokens) == len(messages) + 1:
        message_tokens = [message_tokens[0] + message_tokens[1]] + message_tokens[2:]

    if len(message_tokens) != len(messages):
        return {
            "token_length": 0,
            "turn_count": len(messages),
            "supervised_token_length": 0,
            "input_ids": [],
            "shift_labels": [],
        }

    input_ids = []
    same_position_labels = []
    for role_msg, tokens in zip(messages, message_tokens):
        input_ids.extend(tokens)
        if role_msg["role"] == "assistant":
            same_position_labels.extend(tokens)
        else:
            same_position_labels.extend([-100] * len(tokens))

    if max_input_length is not None and max_input_length > 0 and len(input_ids) > max_input_length:
        input_ids = input_ids[-max_input_length:]
        same_position_labels = same_position_labels[-max_input_length:]

    shift_labels = next_token_assistant_labels(input_ids, same_position_labels)
    supervised_token_length = sum(label != -100 for label in shift_labels)
    return {
        "token_length": len(input_ids),
        "turn_count": len(messages),
        "supervised_token_length": supervised_token_length,
        "input_ids": input_ids,
        "shift_labels": shift_labels,
    }


class SambaYDatasetBase(ABC):
    """Base class for SambaY-native decoder-only multi-turn datasets."""

    name = None
    max_token_length = DEFAULT_MAX_TOKEN_LENGTH

    def __init__(
        self,
        model_name,
        split="train",
        max_input_length=None,
        force_reprocess=False,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length or self.max_token_length
        self._init_tokenizer()
        cached_path = self.cache_path
        if not force_reprocess and os.path.exists(cached_path):
            print(f"Loading SambaY dataset from local path: {cached_path}")
            self.data = load_from_disk(cached_path)
            if len(self.data) == 0:
                print(f"Cached SambaY dataset is empty, rebuilding: {cached_path}")
                self._init_data(split)
        else:
            self._init_data(split)

    @property
    def cache_path(self):
        model_key = self.model_name.replace("/", "_")
        return os.path.join(HF_HOME, "datasets", f"sambay_{self.name}_{model_key}")

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<PAD>"
            self.tokenizer.pad_token_id = 128004

    def _messages_to_sambay_sample(self, messages):
        return messages_to_sambay_sample(
            messages,
            self.tokenizer,
            max_input_length=self.max_input_length,
        )

    def _filter_valid_sample(self, item):
        return (
            item["token_length"] > 1
            and item["token_length"] <= self.max_input_length
            and item["supervised_token_length"] > 0
        )

    @abstractmethod
    def _init_data(self, split):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

