"""YOCO-native wrappers for the repository SFT datasets."""

from datasets import load_dataset

from baselines.YOCO.data.base import (
    CPU_NUM,
    HF_HOME,
    YOCO_FEATURES,
    YOCODatasetBase,
    normalize_agentlans_messages,
)


class UltraChatYOCODataset(YOCODatasetBase):
    name = "ultrachat_200k"

    def _init_data(self, split):
        self.data = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
        remove_columns = [col for col in ["prompt", "prompt_id"] if col in self.data.column_names]
        if remove_columns:
            self.data = self.data.remove_columns(remove_columns)

        self.data = self.data.map(
            lambda example: self._messages_to_yoco_sample(example["messages"]),
            remove_columns=["messages"],
            features=YOCO_FEATURES,
            num_proc=CPU_NUM,
        )
        self.data = self.data.filter(self._filter_valid_sample, num_proc=CPU_NUM)
        self.data.save_to_disk(self.cache_path)


class AgentLansYOCODataset(YOCODatasetBase):
    hf_config_name = None

    def _init_data(self, split):
        self.data = load_dataset("agentlans/multiturn-chat", self.hf_config_name, split=split)
        remove_columns = [col for col in ["source"] if col in self.data.column_names]
        if remove_columns:
            self.data = self.data.remove_columns(remove_columns)

        def convert(example):
            messages = normalize_agentlans_messages(example["conversations"])
            return self._messages_to_yoco_sample(messages)

        self.data = self.data.map(
            convert,
            remove_columns=["conversations"],
            features=YOCO_FEATURES,
            num_proc=CPU_NUM,
        )
        self.data = self.data.filter(self._filter_valid_sample, num_proc=CPU_NUM)
        self.data.save_to_disk(self.cache_path)


class ChatsV2YOCODataset(AgentLansYOCODataset):
    name = "chats_v2_k200000"
    hf_config_name = "chats_v2_k200000"


class WildChatYOCODataset(AgentLansYOCODataset):
    name = "allenai-WildChat-4.8M"
    hf_config_name = "allenai-WildChat-4.8M"


class ClaudeYOCODataset(AgentLansYOCODataset):
    name = "claude"
    hf_config_name = "claude"


class InfinityInstructYOCODataset(AgentLansYOCODataset):
    name = "infinity-instruct"
    hf_config_name = "infinity-instruct"


class MagpieUltraYOCODataset(AgentLansYOCODataset):
    name = "magpie-ultra"
    hf_config_name = "magpie-ultra"
