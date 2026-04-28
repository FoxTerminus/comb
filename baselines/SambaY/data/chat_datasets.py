"""SambaY-native wrappers for raw multi-turn chat datasets."""

from datasets import load_dataset

from baselines.SambaY.data.base import (
    CPU_NUM,
    SAMBAY_FEATURES,
    SambaYDatasetBase,
    normalize_agentlans_messages,
)


class UltraChatSambaYDataset(SambaYDatasetBase):
    name = "ultrachat_200k"

    def _init_data(self, split):
        self.data = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
        remove_columns = [col for col in ["prompt", "prompt_id"] if col in self.data.column_names]
        if remove_columns:
            self.data = self.data.remove_columns(remove_columns)

        self.data = self.data.map(
            lambda example: self._messages_to_sambay_sample(example["messages"]),
            remove_columns=["messages"],
            features=SAMBAY_FEATURES,
            num_proc=CPU_NUM,
        )
        self.data = self.data.filter(self._filter_valid_sample, num_proc=CPU_NUM)
        self.data.save_to_disk(self.cache_path)


class AgentLansSambaYDataset(SambaYDatasetBase):
    hf_config_name = None

    def _init_data(self, split):
        self.data = load_dataset("agentlans/multiturn-chat", self.hf_config_name, split=split)
        remove_columns = [col for col in ["source"] if col in self.data.column_names]
        if remove_columns:
            self.data = self.data.remove_columns(remove_columns)

        def convert(example):
            messages = normalize_agentlans_messages(example["conversations"])
            return self._messages_to_sambay_sample(messages)

        self.data = self.data.map(
            convert,
            remove_columns=["conversations"],
            features=SAMBAY_FEATURES,
            num_proc=CPU_NUM,
        )
        self.data = self.data.filter(self._filter_valid_sample, num_proc=CPU_NUM)
        self.data.save_to_disk(self.cache_path)


class ChatsV2SambaYDataset(AgentLansSambaYDataset):
    name = "chats_v2_k200000"
    hf_config_name = "chats_v2_k200000"


class WildChatSambaYDataset(AgentLansSambaYDataset):
    name = "allenai-WildChat-4.8M"
    hf_config_name = "allenai-WildChat-4.8M"


class ClaudeSambaYDataset(AgentLansSambaYDataset):
    name = "claude"
    hf_config_name = "claude"


class InfinityInstructSambaYDataset(AgentLansSambaYDataset):
    name = "infinity-instruct"
    hf_config_name = "infinity-instruct"


class MagpieUltraSambaYDataset(AgentLansSambaYDataset):
    name = "magpie-ultra"
    hf_config_name = "magpie-ultra"

