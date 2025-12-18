"""This file distills the dataset using mimo-v2-flash."""

import os
import openai
from openai import OpenAI

from datasets import Dataset
from data import DATASET_DICT
from data.base import HF_HOME

client = OpenAI(
    api_key=os.environ.get("MIMO_API_KEY"),
    base_url="https://api.xiaomimimo.com/v1"
)

def distill_dataset(model_name="XiaomiMiMo/MiMo-V2-Flash"):
    for name, dataset in DATASET_DICT.items():
        print(f"Distilling dataset: {name}")
        split = "train_sft" if name == "ultrachat_200k" else "train"
        ds = dataset(model_name, split=split).data
        print(f"Dataset size: {len(ds)}")
        ds_new = []
        count = 0
        for sample in ds:
            try:
                complition = client.chat.completions.create(
                    model="mimo-v2-flash",
                    messages=sample["messages"][:-1],
                    max_completion_tokens=1024,
                    temperature=0.8,
                    top_p=0.95,
                    stream=False,
                    stop=None,
                    frequency_penalty=0,
                    presence_penalty=0,
                    extra_body={
                        "thinking": {"type": "disabled"}
                    }
                )
                sample["messages"][-1]["content"] = complition.choices[0].message.content
            except openai.APIStatusError as e:
                print(f"APIStatusError: {e}, using original response.")
            finally:
                ds_new.append(sample)
                count += 1
                if count % 10000 == 0:
                    print(f"Processed samples: {count}/{len(ds)}")
        
        distilled_dataset = Dataset.from_list(ds_new)
        distilled_dataset.save_to_disk(HF_HOME + f'/datasets/{name}_{model_name.replace("/", "_")}')

if __name__ == "__main__":
    model_name = "XiaomiMiMo/MiMo-V2-Flash"
    distill_dataset(model_name)