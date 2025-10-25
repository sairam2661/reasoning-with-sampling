import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_lengths(csv_path):
    """Analyze completion lengths in a CSV file."""
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print(f"File: {csv_path.name}")
    print(f"{'='*80}")
    
    # Analyze each type of completion
    for col in ['std_completion', 'naive_completion', 'mcmc_completion']:
        if col not in df.columns:
            continue
            
        # Character lengths
        lengths = df[col].fillna('').astype(str).str.len()
        
        print(f"\n{col.upper()}:")
        print(f"  Character length stats:")
        print(f"    Mean:   {lengths.mean():.1f}")
        print(f"    Median: {lengths.median():.1f}")
        print(f"    Min:    {lengths.min():.1f}")
        print(f"    Max:    {lengths.max():.1f}")
        print(f"    95th percentile: {lengths.quantile(0.95):.1f}")
        print(f"    99th percentile: {lengths.quantile(0.99):.1f}")
        
        # Line count
        line_counts = df[col].fillna('').astype(str).apply(lambda x: len(x.split('\n')))
        print(f"  Line count stats:")
        print(f"    Mean:   {line_counts.mean():.1f}")
        print(f"    Median: {line_counts.median():.1f}")
        print(f"    Max:    {line_counts.max():.1f}")
        
        # Rough token estimate (tokens ≈ chars / 4 for code)
        token_estimate = lengths / 4
        print(f"  Estimated tokens (chars/4):")
        print(f"    Mean:   {token_estimate.mean():.1f}")
        print(f"    Median: {token_estimate.median():.1f}")
        print(f"    95th percentile: {token_estimate.quantile(0.95):.1f}")
        print(f"    99th percentile: {token_estimate.quantile(0.99):.1f}")
        print(f"    Max:    {token_estimate.max():.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="results/phi/he",
        help="Folder containing CSV result files."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for result CSVs (default: *.csv)."
    )
    args = parser.parse_args()
    
    results_dir = Path(args.folder)
    csv_files = sorted(results_dir.glob(args.pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {args.pattern} in {results_dir}")
    
    # Aggregate statistics across all files
    all_std_lengths = []
    all_naive_lengths = []
    all_mcmc_lengths = []
    
    for f in csv_files:
        df = pd.read_csv(f)
        
        if 'std_completion' in df.columns:
            all_std_lengths.extend(df['std_completion'].fillna('').astype(str).str.len().tolist())
        if 'naive_completion' in df.columns:
            all_naive_lengths.extend(df['naive_completion'].fillna('').astype(str).str.len().tolist())
        if 'mcmc_completion' in df.columns:
            all_mcmc_lengths.extend(df['mcmc_completion'].fillna('').astype(str).str.len().tolist())
        
        analyze_lengths(f)
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL FILES")
    print(f"{'='*80}")
    
    if all_std_lengths:
        std_tokens = np.array(all_std_lengths) / 4
        print(f"\nSTD_COMPLETION (n={len(all_std_lengths)}):")
        print(f"  Estimated tokens: Mean={std_tokens.mean():.1f}, Median={np.median(std_tokens):.1f}, Max={std_tokens.max():.1f}")
        print(f"  95th percentile: {np.percentile(std_tokens, 95):.1f} tokens")
        print(f"  99th percentile: {np.percentile(std_tokens, 99):.1f} tokens")
    
    if all_naive_lengths:
        naive_tokens = np.array(all_naive_lengths) / 4
        print(f"\nNAIVE_COMPLETION (n={len(all_naive_lengths)}):")
        print(f"  Estimated tokens: Mean={naive_tokens.mean():.1f}, Median={np.median(naive_tokens):.1f}, Max={naive_tokens.max():.1f}")
        print(f"  95th percentile: {np.percentile(naive_tokens, 95):.1f} tokens")
        print(f"  99th percentile: {np.percentile(naive_tokens, 99):.1f} tokens")
    
    if all_mcmc_lengths:
        mcmc_tokens = np.array(all_mcmc_lengths) / 4
        print(f"\nMCMC_COMPLETION (n={len(all_mcmc_lengths)}):")
        print(f"  Estimated tokens: Mean={mcmc_tokens.mean():.1f}, Median={np.median(mcmc_tokens):.1f}, Max={mcmc_tokens.max():.1f}")
        print(f"  95th percentile: {np.percentile(mcmc_tokens, 95):.1f} tokens")
        print(f"  99th percentile: {np.percentile(mcmc_tokens, 99):.1f} tokens")
        print(f"  NOTE: MCMC includes the full prompt, so actual completion is shorter")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    if all_std_lengths and all_naive_lengths:
        combined = np.array(all_std_lengths + all_naive_lengths) / 4
        p99 = np.percentile(combined, 99)
        recommended = int(np.ceil(p99 / 100) * 100)  # Round up to nearest 100
        print(f"Based on 99th percentile of std/naive completions: {p99:.1f} tokens")
        print(f"Recommended max_new_tokens: {recommended} (covers 99% of cases)")
        print(f"Current setting (3072) is {'sufficient' if 3072 >= recommended else 'too low'}")

if __name__ == "__main__":
    main()