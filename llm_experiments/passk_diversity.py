import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from grader_utils.math_grader import grade_answer
import itertools


# ---- Input Args ----
RESULTS_DIR = "results/qwen_math"
PREFIX = "qwen_math_rlcomp_base_low_temp_results_10_0.25_{idx}_{seed}.csv"
SEEDS = list(range(0, 16))        # seeds 0..10 inclusive
IDXS = list(range(0, 5))          # 5 shards
# -------------------------------------

def safe_grade(mcmc_ans, correct_ans):
    try:
        return int(grade_answer(mcmc_ans, correct_ans))
    except Exception:
        return 0


def plot_passk():
  
  rows_per_idx = []
  for idx in IDXS:
      f = os.path.join(RESULTS_DIR, PREFIX.format(seed=0, idx=idx))
      df = pd.read_csv(f)
      rows_per_idx.append(len(df))
  offsets = np.cumsum([0] + rows_per_idx[:-1])  # global offsets per shard
  TOTAL = sum(rows_per_idx)                      # total problems (should be 500)
  
  correct_by_seed = np.zeros((max(SEEDS) + 1, TOTAL), dtype=np.uint8)
  
  for seed in SEEDS:
      for shard_i, idx in enumerate(IDXS):
          f = os.path.join(RESULTS_DIR, PREFIX.format(seed=seed, idx=idx))
          df = pd.read_csv(f)
  
          # Sanity: if a shard length differs, align on the min length
          L = min(rows_per_idx[shard_i], len(df))
          base = offsets[shard_i]
  
          # Fill correctness for this seed on the global problem indices
          for i in range(L):
              prob_id = base + i
              correct_by_seed[seed, prob_id] = safe_grade(df["mcmc_answer"][i], df["correct_answer"][i])
  
  # Compute best-of-N accuracies for N=0..10 (inclusive)
  best_of_N_acc = []
  for N in range(0, max(SEEDS) + 1):
      # Seeds considered: 0..N
      # Best-of-N correctness is OR across those seeds (per problem)
      best_correct = correct_by_seed[0:N+1, :].max(axis=0)
      acc = best_correct.mean()  # fraction correct
      best_of_N_acc.append(acc)
  
  # Print and plot
  for N, acc in enumerate(best_of_N_acc):
      print(f"Best-of-{N}: {acc:.4f}")
  
  
  
  best_of_N_acc = []
  
  num_seeds = len(SEEDS)
  
  for N in range(1, num_seeds + 1):
      accs = []
      # Enumerate all subsets if small, otherwise sample random subsets
      if num_seeds <= 15 and N <= 5:  # exact enumeration feasible
          for subset in itertools.combinations(SEEDS, N):
              subset_correct = correct_by_seed[list(subset), :].max(axis=0)
              accs.append(subset_correct.mean())
      else:  # fallback: random sampling if too many combinations
          n_samples = 200  # tune this
          rng = np.random.default_rng(0)
          for _ in range(n_samples):
              subset = rng.choice(SEEDS, size=N, replace=False)
              subset_correct = correct_by_seed[subset, :].max(axis=0)
              accs.append(subset_correct.mean())
  
      best_of_N_acc.append((N, np.mean(accs)))  # only store mean
  
  
  for N, mean_acc in best_of_N_acc:
      print(f"Best-of-{N}: {mean_acc:.4f}")
  
  # Plot (no error bars)
  plt.figure(figsize=(6,4))
  plt.plot(
      [N for N, _ in best_of_N_acc],
      [mean for _, mean in best_of_N_acc],
      "o-"
  )
  
  plt.xlabel("k")
  plt.ylabel("Pass@k Accuracy")
  plt.title("MATH500")
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()
  plt.show()
