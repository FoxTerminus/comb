"""This file distills the dataset using mimo-v2-flash."""

import os
import openai
from openai import OpenAI
from datasets import Dataset
from data import DATASET_DICT
from data.base import HF_HOME
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock

class RateLimiter:
    def __init__(self, max_requests_per_minute=50):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 60 seconds
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # After sleeping, clean again and continue
                self.requests = [req_time for req_time in self.requests if time.time() - req_time < 60]

            self.requests.append(time.time())

def token_count(tokenizer, messages):
    total_tokens = 0
    for msg in messages:
        total_tokens += len(tokenizer.encode(msg["content"]))
    return total_tokens

def process_single_sample(sample, tokenizer, client, rate_limiter, model_name):
    """Process a single sample with rate limiting."""
    try:
        # Wait if we're hitting rate limits
        rate_limiter.wait_if_needed()

        completion = client.chat.completions.create(
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
        sample["messages"][-1]["content"] = completion.choices[0].message.content
    except openai.APIStatusError as e:
        # print(f"APIStatusError: {e}, using original response.")
        pass
    except Exception as e:
        print(f"Unexpected error: {e} Using original response.")

    # Return the sample along with token info for statistics
    token_len = token_count(tokenizer, sample["messages"])
    return sample, token_len

def distill_dataset(model_name, max_workers=4, requests_per_minute=50):
    """
    Distill dataset with parallel processing and rate limiting.

    Args:
        model_name: The model to use for distillation
        max_workers: Number of parallel threads (default: 4)
        requests_per_minute: RPM limit for the API (default: 50)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    client = OpenAI(
        api_key=os.environ.get("MIMO_API_KEY"),
        base_url="https://api.xiaomimimo.com/v1"
    )

    for name, dataset in DATASET_DICT.items():
        print(f"Distilling dataset: {name}")
        split = "train_sft" if name == "ultrachat_200k" else "train"
        ds = dataset(model_name, split=split).data
        print(f"Dataset size: {len(ds)}")

        # Create rate limiter
        rate_limiter = RateLimiter(max_requests_per_minute=requests_per_minute)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    process_single_sample,
                    sample,
                    tokenizer,
                    client,
                    rate_limiter,
                    model_name
                ): sample for sample in ds
            }

            # Collect results as they complete
            ds_new = []
            count = 0
            long_count = 0

            for future in as_completed(future_to_sample):
                sample, token_len = future.result()
                ds_new.append(sample)
                count += 1
                if token_len > 1024:
                    long_count += 1

                # Print progress periodically
                if count % 10000 == 0:
                    print(f"Processed {count}/{len(ds)} samples")

        # Save the distilled dataset
        print(f"Long samples: {long_count}/{count}")
        distilled_dataset = Dataset.from_list(ds_new)
        save_path = HF_HOME + f'/datasets/{name}_{model_name.replace("/", "_")}'
        distilled_dataset.save_to_disk(save_path)
        print(f"Saved distilled dataset to: {save_path}")

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    MAX_WORKERS = 20  # Number of parallel threads
    REQUESTS_PER_MINUTE = 100  # Adjust based on your API's RPM limit

    distill_dataset(model_name, max_workers=MAX_WORKERS, requests_per_minute=REQUESTS_PER_MINUTE)
