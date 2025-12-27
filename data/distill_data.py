"""distill_datasets.py
This file distills datasets using mimo-v2-flash with parallel processing and rate limiting.
"""

import os
import openai
from openai import OpenAI
from datasets import Dataset
from data import DATASET_DICT
from data.base import HF_HOME, CPU_NUM
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
import argparse

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

    return sample

def distill_single_dataset(name, dataset_func, model_name, max_workers=4, requests_per_minute=50):
    """
    Distill a single dataset with parallel processing and rate limiting.
    
    Args:
        name: name of the dataset
        dataset_func: function to load the dataset
        model_name: name of the model to use for tokenization and distillation
        max_workers: number of parallel workers
        requests_per_minute: API request limit
    """
    print(f"\n{'='*60}")
    print(f"Distilling dataset: {name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    client = OpenAI(
        api_key=os.environ.get("MIMO_API_KEY"),
        base_url="https://api.xiaomimimo.com/v1"
    )
    
    split = "train_sft" if name == "ultrachat_200k" else "train"
    ds = dataset_func(model_name, split=split).data
    print(f"Dataset size: {len(ds):,} samples")
    
    # Create rate limiter
    rate_limiter = RateLimiter(max_requests_per_minute=requests_per_minute)
    
    # Use ThreadPoolExecutor for parallel processing
    print("Starting distillation process...")
    start_time = time.time()
    
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
        
        for future in as_completed(future_to_sample):
            sample = future.result()
            ds_new.append(sample)
            count += 1
            
            # Print progress periodically
            if count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"   Processed {count}/{len(ds)} samples ({count/elapsed:.1f} samples/sec)")
    
    elapsed_time = time.time() - start_time
    print(f"Distillation completed: {count} samples processed")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {count/elapsed_time:.1f} samples/second")
    
    # Save the distilled dataset
    print("Saving distilled dataset...")
    distilled_dataset = Dataset.from_list(ds_new)
    save_path = HF_HOME + f'/datasets/{name}_{model_name.replace("/", "_")}_distilled'
    distilled_dataset.save_to_disk(save_path)
    print(f"Saved distilled dataset to: {save_path}")
    
    return ds_new

def distill_datasets_sequentially(model_name, dataset_names=None, max_workers=4, requests_per_minute=50):
    """
    Distill datasets sequentially.
    
    Args:
        model_name: name of the model to use for distillation
        dataset_names: the datasets to be distilled (default: all datasets)
        max_workers: number of parallel workers
        requests_per_minute: API request limit
    """
    print("\n" + "="*70)
    print("DISTILLING DATASETS")
    print("="*70)
    
    all_results = {}
    
    # Determine which datasets to process
    if dataset_names is None:
        dataset_items = DATASET_DICT.items()
        print(f"Will distill all {len(DATASET_DICT)} datasets")
    else:
        dataset_items = [(name, DATASET_DICT[name]) for name in dataset_names if name in DATASET_DICT]
        print(f"Will distill {len(dataset_items)} specified datasets")
    
    # Process each dataset sequentially
    total_start_time = time.time()
    
    for name, dataset_func in dataset_items:
        try:
            print(f"\nStarting dataset {name}...")
            dataset_start_time = time.time()
            
            distilled_data = distill_single_dataset(
                name, dataset_func, model_name, max_workers, requests_per_minute
            )
            
            dataset_time = time.time() - dataset_start_time
            all_results[name] = {
                "distilled_data": distilled_data,
                "time_taken": dataset_time,
                "samples": len(distilled_data)
            }
            
            print(f"✅ Dataset {name} completed in {dataset_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error processing dataset {name}: {e}")
            print("Skipping to next dataset...")
            continue
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("DISTILLATION COMPLETED")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successfully distilled {len(all_results)} datasets")
    print("="*70)
    
    # Print summary
    if all_results:
        print("\nDISTILLATION SUMMARY:")
        print("-" * 80)
        print(f"{'Dataset':<25} {'Samples':<10} {'Time (s)':<10} {'Speed (samples/s)':<20}")
        print("-" * 80)
        
        for name, result in all_results.items():
            speed = result["samples"] / result["time_taken"] if result["time_taken"] > 0 else 0
            print(f"{name:<25} {result['samples']:<10,} {result['time_taken']:<10.1f} {speed:<20.1f}")
        print("-" * 80)
    
    return all_results

def main():
    """Main function for standalone dataset distillation"""
    parser = argparse.ArgumentParser(description='Distill datasets using mimo-v2-flash')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name for distillation (default: meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Specific datasets to distill (default: all datasets)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU_NUM * 2, max 20)')
    parser.add_argument('--requests_per_minute', type=int, default=100,
                       help='API request limit per minute (default: 100)')
    parser.add_argument('--check_api_key', action='store_true',
                       help='Check if MIMO_API_KEY is set before proceeding')
    
    args = parser.parse_args()
    
    # Check API key if requested
    if args.check_api_key:
        api_key = os.environ.get("MIMO_API_KEY")
        if not api_key:
            print("❌ Error: MIMO_API_KEY environment variable is not set!")
            print("Please set it with: export MIMO_API_KEY='your_api_key_here'")
            return
        print(f"✅ MIMO_API_KEY is set (length: {len(api_key)} characters)")
    
    # Determine optimal number of workers
    if args.max_workers is None:
        max_workers = min(20, CPU_NUM * 2)
    else:
        max_workers = args.max_workers
    
    print(f"Starting distillation with model: {args.model}")
    print(f"System has {CPU_NUM} CPU cores available")
    print(f"Using {max_workers} workers for parallel processing")
    print(f"API rate limit: {args.requests_per_minute} requests per minute")
    
    if args.datasets:
        print(f"Datasets to distill: {', '.join(args.datasets)}")
    else:
        print("Will distill all available datasets")
    
    # Ask for confirmation
    confirmation = input("\nDo you want to proceed with distillation? This may incur API costs. (y/n): ")
    if confirmation.lower() != 'y':
        print("Distillation cancelled.")
        return
    
    # Start distillation
    results = distill_datasets_sequentially(
        model_name=args.model,
        dataset_names=args.datasets,
        max_workers=max_workers,
        requests_per_minute=args.requests_per_minute
    )
    
    if results:
        print(f"\n✅ Successfully distilled {len(results)} datasets")
        print(f"All distilled datasets saved to: {HF_HOME}/datasets/")
    else:
        print("\n❌ No datasets were successfully distilled")

if __name__ == "__main__":
    main()
