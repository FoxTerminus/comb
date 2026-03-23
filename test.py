from torch.testing import assert_close
from transformers import LlamaConfig

from models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration

# Initialize the model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = CombLlamaForConditionalGeneration(from_scratch=True,
            config=CombLlamaConfig(LlamaConfig.from_pretrained(model_name)))

chunk_ids = [[128000, 512, 128009], [128000, 512, 128009]]
input_ids = [128000, 512, 128009]

result1 = model(input_ids, chunk_ids)

cu_seqlens_q = []
result2 = model(input_ids, chunk_ids, cu_seqlens_q=cu_seqlens_q)

assert_close(result1.logits, result2.logits)