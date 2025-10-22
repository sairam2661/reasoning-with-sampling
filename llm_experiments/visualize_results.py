import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from grader_utils.math_grader import grade_answer
import numpy as np 

def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0

def load_and_evaluate(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'proposal_type' in df.columns:
        proposal_type = df['proposal_type'].iloc[0]
    else:
        # Extract from filename
        if 'uniform' in str(csv_path):
            proposal_type = 'uniform'
        elif 'priority' in str(csv_path):
            proposal_type = 'priority'
        elif 'restart' in str(csv_path):
            proposal_type = 'restart'
        else:
            proposal_type = 'unknown'
    
    # Grade each answer
    df['std_correct'] = [safe_grade(ans, correct) for ans, correct in 
                         zip(df['std_answer'], df['correct_answer'])]
    df['mcmc_correct'] = [safe_grade(ans, correct) for ans, correct in 
                          zip(df['mcmc_answer'], df['correct_answer'])]
    
    return df, proposal_type

def visualize_comparison(results_dir):    
    results_dir = Path(results_dir)
    csv_files = sorted(results_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    # Load all results
    all_data = []
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}...")
        df, proposal_type = load_and_evaluate(csv_file)
        df['proposal'] = proposal_type
        all_data.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    
    methods = []
    accuracies = []
    
    methods.append('Base')
    accuracies.append(combined_df['std_correct'].mean())
    
    proposals = sorted(combined_df['proposal'].unique())
    for proposal in proposals:
        df_proposal = combined_df[combined_df['proposal'] == proposal]
        methods.append(proposal.capitalize())
        accuracies.append(df_proposal['mcmc_correct'].mean())
    
    colors = ['#7EB6D9', '#5B9BD5', '#2E5C8A']  # Change later
    if len(methods) > 3:
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(methods)))
    else:
        colors = colors[:len(methods)]
    
    # Create bar chart
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=13)
    ax1.set_title('Method Comparison (MATH 500)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim([0, min(max(accuracies) * 1.12, 1.0)])
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=0, labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.set_axisbelow(True)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = axes[1]
    
    if 'acceptance_ratio' in combined_df.columns:
        proposals_list = sorted(combined_df['proposal'].unique())
        acceptance_data = [combined_df[combined_df['proposal'] == p]['acceptance_ratio'].dropna() 
                          for p in proposals_list]
        
        bp = ax2.boxplot(acceptance_data, labels=[p.capitalize() for p in proposals_list], 
                        patch_artist=True, showmeans=True,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='darkred'))
        
        box_colors = plt.cm.Blues(np.linspace(0.5, 0.8, len(proposals_list)))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        for i, (proposal, data) in enumerate(zip(proposals_list, acceptance_data)):
            mean_val = data.mean()
            ax2.text(i+1, mean_val, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Acceptance Ratio', fontsize=13)
        ax2.set_title('MCMC Acceptance Ratios', fontsize=15, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.tick_params(labelsize=11)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, 'No acceptance ratio data', 
                ha='center', va='center', fontsize=12)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
    
    ax3 = axes[2]
    
    cumulative_base = combined_df.groupby(combined_df.index)['std_correct'].first().expanding().mean()
    ax3.plot(cumulative_base.index, cumulative_base, 
            label='Base', linewidth=2.5, linestyle='--', color='#7EB6D9', alpha=0.9)
    
    colors_mcmc = plt.cm.Blues(np.linspace(0.5, 0.9, len(proposals)))
    for i, proposal in enumerate(proposals):
        df_proposal = combined_df[combined_df['proposal'] == proposal].sort_index()
        cumulative_acc = df_proposal['mcmc_correct'].expanding().mean()
        ax3.plot(cumulative_acc.index, cumulative_acc, 
                label=proposal.capitalize(), linewidth=2.5, color=colors_mcmc[i])
    
    ax3.set_xlabel('Number of Problems', fontsize=13)
    ax3.set_ylabel('Cumulative Accuracy', fontsize=13)
    ax3.set_title('Cumulative Accuracy Over Time', fontsize=15, fontweight='bold', pad=15)
    ax3.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.tick_params(labelsize=11)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = results_dir / 'comparison_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    output_pdf = results_dir / 'comparison_visualization.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nBase (standard sampling, temp=1.0):")
    print(f"  Accuracy: {combined_df['std_correct'].mean():.3f} ({combined_df['std_correct'].sum()}/{len(combined_df)})")
    
    print(f"\nMCMC Results by Proposal:")
    for proposal in sorted(combined_df['proposal'].unique()):
        df_proposal = combined_df[combined_df['proposal'] == proposal]
        base_acc = df_proposal['std_correct'].mean()
        mcmc_acc = df_proposal['mcmc_correct'].mean()
        improvement = (mcmc_acc - base_acc) * 100
        
        print(f"  {proposal.capitalize()}: {mcmc_acc:.3f} ({df_proposal['mcmc_correct'].sum()}/{len(df_proposal)})", end="")
        print(f" | Improvement over Base: {improvement:+.1f}%", end="")
        if 'acceptance_ratio' in df_proposal.columns:
            print(f" | Acceptance: {df_proposal['acceptance_ratio'].mean():.3f}")
        else:
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="llm_experiments/results/qwen_math/",
                       help="Folder containing CSV result files")
    args = parser.parse_args()
    
    visualize_comparison(args.folder)