import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from grader_utils.math_grader import grade_answer

def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0

def ensure_correctness_columns(df):
    if "std_correct" not in df.columns:
        df["std_correct"] = [
            safe_grade(a, c) for a, c in zip(df.get("std_answer", []), df.get("correct_answer", []))
        ]
    if "mcmc_correct" not in df.columns:
        df["mcmc_correct"] = [
            safe_grade(a, c) for a, c in zip(df.get("mcmc_answer", []), df.get("correct_answer", []))
        ]
    return df

def infer_proposal_from_filename(path):
    name = str(path).lower()
    if "uniform" in name:
        return "uniform"
    if "priority" in name:
        return "priority"
    if "restart" in name:
        return "restart"
    return "unknown"

def load_one_csv(csv_path):
    df = pd.read_csv(csv_path)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    if "proposal_type" in df.columns and pd.notna(df["proposal_type"]).all():
        proposal = df["proposal_type"].astype(str).str.lower()
        df["proposal"] = proposal
    else:
        df["proposal"] = infer_proposal_from_filename(csv_path)

    if "problem_idx" not in df.columns:
        raise ValueError(f"'problem_idx' column is missing in {csv_path}")

    df = ensure_correctness_columns(df)

    keep_cols = [
        "problem_idx", "correct_answer",
        "std_answer", "std_correct",
        "mcmc_answer", "mcmc_correct",
        "proposal"
    ]
    for opt in ("acceptance_ratio",):
        if opt in df.columns:
            keep_cols.append(opt)

    return df[keep_cols].copy()


def aggregate_for_plot(all_df):
    methods, accuracies, counts = [], [], []

    base_df = (all_df
               .sort_values(["problem_idx"])
               .drop_duplicates(subset=["problem_idx"], keep="first"))
    base_acc = base_df["std_correct"].mean() if not base_df.empty else 0.0
    base_n = int(base_df.shape[0])
    methods.append("Base")
    accuracies.append(base_acc)
    counts.append(base_n)

    proposals = sorted([p for p in all_df["proposal"].dropna().unique()])
    for p in proposals:
        sub = (all_df[all_df["proposal"] == p]
               .sort_values(["problem_idx"])
               .drop_duplicates(subset=["problem_idx"], keep="first"))
        acc = sub["mcmc_correct"].mean() if not sub.empty else 0.0
        n = int(sub.shape[0])
        methods.append(p.capitalize())
        accuracies.append(acc)
        counts.append(n)

    return methods, accuracies, counts

def plot_bar(methods, accuracies, counts, title, out_png):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x = range(len(methods))
    bars = ax.bar(x, accuracies, edgecolor="black", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylim(0.0, min(1.0, max(accuracies + [0.0]) * 1.12 if accuracies else 1.0))
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for rect, acc, n in zip(bars, accuracies, counts):
        h = rect.get_height()
        label = f"{acc*100:.1f}% · n={n}"
        ax.text(rect.get_x() + rect.get_width() / 2.0,
                h + 0.01,
                label,
                ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="llm_experiments/results/qwen_math/math",
        help="Folder containing CSV result files (batched)."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for result CSVs (default: *.csv)."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Method Comparison (MATH 500)",
        help="Plot title."
    )
    args = parser.parse_args()

    results_dir = Path(args.folder)
    csv_files = sorted(results_dir.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {args.pattern} in {results_dir}")

    frames = []
    for f in csv_files:
        print(f"Loading {f.name} ...")
        df = load_one_csv(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    methods, accuracies, counts = aggregate_for_plot(combined)
    
    out_png = results_dir / "comparison_bar.png"
    plot_bar(methods, accuracies, counts, args.title, out_png)

if __name__ == "__main__":
    main()
