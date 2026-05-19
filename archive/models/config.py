"""Comb configuration for Qwen3 backbone."""

from typing import List, Optional, Union

from transformers import Qwen3Config
from transformers.configuration_utils import PretrainedConfig


class CombConfig(PretrainedConfig):
    """Configuration for Comb: chunk encoder + cross-attention augmented Qwen3.

    cross_attention_layers defaults to ``[3,7,11,15,19,23,27]`` for a
    28-layer backbone (1/4 of layers, evenly spaced, aligned with
    original CombLlama's ``[3,7,11,15,19,23,27,31]`` for 32-layer Llama).
    """

    model_type = "comb_qwen"
    sub_configs = {"text_config": Qwen3Config}

    def __init__(
        self,
        text_config: Optional[Union[Qwen3Config, dict]] = None,
        cross_attention_layers: Optional[List[int]] = None,
        chunk_token_index: int = 151643,
        pad_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = Qwen3Config()
        elif isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)

        self.text_config = text_config
        self.chunk_token_index = chunk_token_index

        if cross_attention_layers is None:
            n = text_config.num_hidden_layers
            self.cross_attention_layers = [3, 7, 11, 15, 19, 23, 27][: n // 4]
        else:
            self.cross_attention_layers = list(cross_attention_layers)

        super().__init__(
            pad_token_id=pad_token_id or text_config.pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_cross_layers(self) -> int:
        return len(self.cross_attention_layers)
