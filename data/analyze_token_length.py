"""analyze_token_lengths.py
This file analyzes the token length distributions of all datasets and generates CDF plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from transformers import AutoTokenizer
from data import DATASET_DICT
from data.base import CPU_NUM
import time

def calculate_token_lengths(dataset, tokenizer, batch_size=1000):
    """Calculate token lengths for each sample in the dataset using map with batch processing"""
    
    print(f"   Calculating token lengths for {len(dataset):,} samples...")
    
    # Determine optimal number of processes
    num_proc = min(CPU_NUM, 16)  # Cap at 16 processes to avoid overhead
    if len(dataset) < 1000:
        num_proc = 1  # For small datasets, use single process to avoid overhead
    elif len(dataset) < 10000:
        num_proc = min(2, CPU_NUM)  # For medium datasets
    
    print(f"   Using {num_proc} processes for parallel processing...")
    
    def batch_token_count(examples):
        """Batch function to count tokens for multiple samples"""
        token_lengths = []
        for messages in examples["messages"]:
            total_tokens = 0
            for msg in messages:
                # Use tokenizer's __call__ method which handles empty strings better
                if msg["content"] and isinstance(msg["content"], str):
                    # Tokenize and count tokens
                    tokens = tokenizer(msg["content"], 
                                      truncation=False, 
                                      padding=False, 
                                      add_special_tokens=False)
                    total_tokens += len(tokens["input_ids"])
            token_lengths.append(total_tokens)
        return {"token_length": token_lengths}
    
    start_time = time.time()
    
    # Apply map function with batch processing
    dataset_with_lengths = dataset.map(
        batch_token_count,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="📊 Calculating token lengths",
        keep_in_memory=False
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract token lengths
    token_lengths = dataset_with_lengths["token_length"]
    
    print(f"   ✅ Token lengths calculated for {len(token_lengths):,} samples")
    print(f"   ⏱️  Time taken: {elapsed_time:.2f} seconds")
    print(f"   📈 Processing speed: {len(dataset)/elapsed_time:.1f} samples/second")
    
    return token_lengths

def plot_cdf(token_lengths, dataset_name, model_name, save_dir="token_length_plots", suffix="original"):
    """Plot CDF of token lengths and save the plot"""
    
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure token_lengths is numpy array
    if not isinstance(token_lengths, np.ndarray):
        token_lengths = np.array(token_lengths)
    
    # Calculate CDF
    sorted_lengths = np.sort(token_lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot CDF line
    plt.plot(sorted_lengths, cdf, linewidth=3, color='blue', label='CDF')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Token Length', fontsize=20)
    plt.ylabel('Cumulative Probability', fontsize=20)
    plt.title(f'Token Length CDF - {dataset_name} ({suffix})\nModel: {model_name}', fontsize=22, pad=20)
    
    # Calculate and plot percentiles
    percentiles = [50, 75, 90, 95, 99]
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    
    for p, color in zip(percentiles, colors):
        percentile_value = np.percentile(token_lengths, p)
        plt.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        plt.text(percentile_value, 0.02, f'{p}%: {percentile_value:.0f}', 
                 rotation=90, verticalalignment='bottom', fontsize=20,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add text box with statistics
    stats_text = (
        f'Total Samples: {len(token_lengths):,}\n'
        f'Mean: {np.mean(token_lengths):.1f}\n'
        f'Median: {np.median(token_lengths):.1f}\n'
        f'Std: {np.std(token_lengths):.1f}\n'
        f'Min: {np.min(token_lengths)}\n'
        f'Max: {np.max(token_lengths)}'
    )
    
    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=13, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set x-axis limit
    x_max = np.percentile(token_lengths, 99) * 1.1
    plt.xlim(0, min(x_max, np.max(token_lengths)))
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save plot (without timestamp)
    filename = f"{dataset_name}_{model_name.replace('/', '_')}_{suffix}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved CDF plot to: {save_path}")
    
    # Return statistics (just for console output, not saved to file)
    stats = {
        "dataset": dataset_name,
        "suffix": suffix,
        "total_samples": len(token_lengths),
        "mean": float(np.mean(token_lengths)),
        "median": float(np.median(token_lengths)),
        "std": float(np.std(token_lengths)),
        "min": int(np.min(token_lengths)),
        "max": int(np.max(token_lengths)),
        "percentiles": {p: float(np.percentile(token_lengths, p)) 
                       for p in [10, 25, 50, 75, 90, 95, 99, 99.9]}
    }
    
    return stats

def analyze_single_dataset(name, dataset_func, model_name, save_dir="token_length_plots", suffix="original", force_recalculate=False):
    """Analyze token length distribution for a single dataset"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing dataset: {name}")
    print(f"{'='*60}")
    
    # Check if plot already exists
    plot_filename = f"{name}_{model_name.replace('/', '_')}_{suffix}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    
    if os.path.exists(plot_path) and not force_recalculate:
        print(f"   Plot already exists: {plot_path}")
        print(f"   Skipping analysis for {name} (use --force to recalculate)")
        return None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    split = "train_sft" if name == "ultrachat_200k" else "train"
    ds = dataset_func(model_name, split=split).data
    print(f"Dataset size: {len(ds):,} samples")
    
    # Calculate token lengths using optimized map function
    token_lengths = calculate_token_lengths(ds, tokenizer)
    
    # Plot CDF
    stats = plot_cdf(token_lengths, name, model_name, save_dir, suffix)
    
    # Print summary
    print(f"\n   Summary for {name}:")
    print(f"   - Mean token length: {stats['mean']:.1f}")
    print(f"   - Median token length: {stats['median']:.1f}")
    print(f"   - 90th percentile: {stats['percentiles'][90]:.0f}")
    print(f"   - 95th percentile: {stats['percentiles'][95]:.0f}")
    print(f"   - 99th percentile: {stats['percentiles'][99]:.0f}")
    
    # Check for long samples
    threshold = 4096
    long_samples = [l for l in token_lengths if l > threshold]
    if long_samples:
        print(f"   ⚠️  Found {len(long_samples)} samples > {threshold} tokens")
        print(f"   Max length: {max(token_lengths)} tokens")
    
    return stats

def analyze_all_datasets(model_name, save_dir="token_length_plots", dataset_names=None, force_recalculate=False):
    """
    Analyze token length distributions for specified datasets
    
    Args:
        model_name: name of the model to use for tokenization
        save_dir: directory to save plots and statistics
        dataset_names: list of dataset names to analyze (None = all datasets)
        force_recalculate: force recalculation even if plot exists
    """
    print("\n" + "="*70)
    print("ANALYZING TOKEN LENGTH DISTRIBUTIONS")
    print(f"Using up to {min(CPU_NUM, 16)} CPU cores for parallel processing")
    if not force_recalculate:
        print("Note: Will skip datasets that already have plots (use --force to recalculate)")
    print("="*70)
    
    all_stats = {}
    
    # Determine which datasets to process
    if dataset_names is None:
        dataset_items = DATASET_DICT.items()
        print(f"Will analyze all {len(DATASET_DICT)} datasets")
    else:
        dataset_items = [(name, DATASET_DICT[name]) for name in dataset_names if name in DATASET_DICT]
        print(f"Will analyze {len(dataset_items)} specified datasets")
    
    processed_count = 0
    skipped_count = 0
    
    # Process each dataset
    for name, dataset_func in dataset_items:
        try:
            stats = analyze_single_dataset(
                name, dataset_func, model_name, save_dir, "original", force_recalculate
            )
            
            if stats is not None:
                all_stats[name] = stats
                processed_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"Error analyzing dataset {name}: {e}")
            print("Skipping to next dataset...")
            continue
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED")
    print(f"Plots saved to: {save_dir}")
    print(f"Processed: {processed_count} datasets, Skipped: {skipped_count} datasets")
    print("="*70)
    
    # Print summary table only for processed datasets
    if all_stats:
        print("\nDATASET SUMMARY TABLE (Processed datasets only):")
        print("-" * 100)
        print(f"{'Dataset':<25} {'Samples':<10} {'Mean':<10} {'Median':<10} {'90th %':<10} {'95th %':<10} {'99th %':<10}")
        print("-" * 100)
        
        for name, stats in all_stats.items():
            print(f"{name:<25} {stats['total_samples']:<10,} {stats['mean']:<10.1f} "
                  f"{stats['median']:<10.1f} {stats['percentiles'][90]:<10.0f} "
                  f"{stats['percentiles'][95]:<10.0f} {stats['percentiles'][99]:<10.0f}")
        print("-" * 100)
    
    return all_stats

def main():
    """Main function for standalone token length analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze token length distributions of datasets')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name for tokenization (default: meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--save_dir', type=str, default='token_length_plots',
                       help='Directory to save plots (default: token_length_plots)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Specific datasets to analyze (default: all datasets)')
    parser.add_argument('--force', action='store_true',
                       help='Force recalculation even if plot already exists')
    
    args = parser.parse_args()
    
    print(f"Starting analysis with model: {args.model}")
    print(f"System has {CPU_NUM} CPU cores available")
    
    start_time = time.time()
    
    # Analyze datasets
    stats = analyze_all_datasets(
        model_name=args.model,
        save_dir=args.save_dir,
        dataset_names=args.datasets,
        force_recalculate=args.force
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.2f} seconds")
    
    if stats:
        print(f"\n✅ Successfully analyzed {len(stats)} datasets")
    else:
        print("\n📊 No new datasets were analyzed (all plots already exist)")

if __name__ == "__main__":
    main()