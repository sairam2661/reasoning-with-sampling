import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict
from grader_utils.he_grader import entry_point as he_entry_point

def extract_code(response, entry_point_name):
    lines = response.split('\n')
    fixed_lines = []
    
    for line in lines:
        if not line.strip():  # Empty line
            fixed_lines.append(line)
            continue
        
        leading_spaces = len(line) - len(line.lstrip())
        content = line.lstrip()
        
        if leading_spaces > 0:
            normalized_indent = ((leading_spaces + 2) // 4) * 4
            fixed_lines.append(' ' * normalized_indent + content)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def load_humaneval_dataset(data_file='data/HumanEval.jsonl'):
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    dataset_dict = {entry["task_id"]: entry for entry in dataset}
    return dataset_dict

def infer_proposal_from_filename(path):
    name = str(path).lower()
    if "uniform" in name:
        return "uniform"
    if "priority" in name:
        return "priority"
    if "restart" in name:
        return "restart"
    return "unknown"

def load_and_merge_csvs(fnames, data_file='data/HumanEval.jsonl'):
    dataset_dict = load_humaneval_dataset(data_file)
    problems = defaultdict(dict)
    
    for fname in fnames:
        print(f"Loading {fname}...")
        df = pd.read_csv(fname)
        
        if "proposal_type" in df.columns and pd.notna(df["proposal_type"].iloc[0]):
            proposal_type = df["proposal_type"].iloc[0].lower()
        else:
            proposal_type = infer_proposal_from_filename(fname)
        
        for _, row in df.iterrows():
            problem_idx = row["problem_idx"]
            task_id = row["id"]
            
            # Get dataset entry
            if task_id not in dataset_dict:
                print(f"Warning: task_id {task_id} not found in dataset")
                continue
            
            dataset_entry = dataset_dict[task_id]
            prompt = dataset_entry["prompt"]
            
            # Store completions
            if problem_idx not in problems:
                problems[problem_idx]['dataset_entry'] = dataset_entry
            
            if 'std' not in problems[problem_idx]:
                problems[problem_idx]['std'] = prompt + row["std_completion"]
            if 'naive' not in problems[problem_idx]:
                problems[problem_idx]['naive'] = prompt + row["naive_completion"]
            
            mcmc_key = f"mcmc_{proposal_type}"
            problems[problem_idx][mcmc_key] = prompt + row["mcmc_completion"]
    
    return problems

def filter_common_problems(problems, methods_to_eval):
    complete_problems = {}
    excluded_problems = []
    
    for problem_idx, problem in problems.items():
        has_all_methods = all(method in problem for method in methods_to_eval)
        
        if has_all_methods:
            complete_problems[problem_idx] = problem
        else:
            missing = [m for m in methods_to_eval if m not in problem]
            excluded_problems.append((problem_idx, missing))
    
    if excluded_problems:
        for prob_idx, missing in excluded_problems[:5]:  # Show first 5
            print(f"  Problem {prob_idx}: missing {missing}")
        if len(excluded_problems) > 5:
            print(f"  ... and {len(excluded_problems) - 5} more")
    
    return complete_problems, len(excluded_problems)

def create_jsonl_for_method(problems, method_key, output_file, data_file):
    dataset_dict = load_humaneval_dataset(data_file)
    all_task_ids = sorted(dataset_dict.keys())
    
    task_id_to_problem = {}
    for problem_idx, problem in problems.items():
        task_id = problem['dataset_entry']['task_id']
        task_id_to_problem[task_id] = problem
    
    with open(output_file, "w") as fout:
        for task_id in all_task_ids:
            if task_id in task_id_to_problem:
                problem = task_id_to_problem[task_id]
                
                if method_key not in problem:
                    code_completion = "    pass  # Method not available"
                else:
                    dataset_entry = problem['dataset_entry']
                    response = problem[method_key]
                    entry_point_name = dataset_entry["entry_point"]
                    
                    code_completion = extract_code(response, entry_point_name)
            else:
                code_completion = "    pass  # Problem not in dataset"
            
            # Write to jsonl
            line = {
                "task_id": task_id,
                "completion": code_completion,
            }
            fout.write(json.dumps(line) + "\n")
    
    print(f"Saved: {output_file} (164 problems total)")
    return output_file

def he_results(fnames, folder, output_fname, data_file='data/HumanEval.jsonl', use_common_subset=True):
    print("Loading and merging CSV files...")
    problems = load_and_merge_csvs(fnames, data_file)
    
    print(f"\nTotal problems loaded: {len(problems)}")
    
    all_methods = set()
    for problem in problems.values():
        all_methods.update(problem.keys())
    all_methods.discard('dataset_entry')
    
    base_methods = ['std', 'naive']
    mcmc_methods = sorted([m for m in all_methods if m.startswith('mcmc_')])
    
    methods_to_eval = [m for m in base_methods if m in all_methods] + mcmc_methods
    
    for method in methods_to_eval:
        count = sum(1 for p in problems.values() if method in p)
        print(f"{method}: {count} problems")
    
    original_problem_count = len(problems)
    if use_common_subset:
        problems, excluded_count = filter_common_problems(problems, methods_to_eval)
        if excluded_count > 0:
            print(f"After filtering: evaluating {len(problems)} common problems")
    
    results = {}
    
    for method in methods_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating {method.upper()}...")
        print(f"{'='*60}")
        
        output_file = f"{folder}/{output_fname}_{method}.jsonl"
        create_jsonl_for_method(problems, method, output_file, data_file)
        
        try:
            result = he_entry_point(output_file)
            
            if use_common_subset and len(problems) < original_problem_count:
                print(f"\nNote: Evaluated on {len(problems)}/164 common problems")
            
            results[method] = result
            print(f"\nResults for {method}: {result}")
        except Exception as e:
            print(f"Error evaluating {method}: {e}")
            results[method] = None
    
    return results

import matplotlib.pyplot as plt

def plot_results(results, output_fname):
    methods = []
    scores = []
    
    method_order = ['std', 'naive', 'mcmc_uniform', 'mcmc_priority', 'mcmc_restart']
    method_labels = {
        'std': 'Std. Sampling (T=1.0)',
        'naive': 'Std. Sampling (T=0.25)',
        'mcmc_uniform': 'MCMC Uniform (10)',
        'mcmc_priority': 'MCMC Priority (10)',
        'mcmc_restart': 'MCMC Restart (10)'
    }
    
    for method in method_order:
        if method in results and results[method] is not None:
            pass_at_1 = results[method].get('pass@1', 0.0)
            methods.append(method_labels.get(method, method))
            scores.append(float(pass_at_1) * 100)
    
    if not methods:
        print("No results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(methods)), scores)
    
    ax.set_ylabel('Pass@1 Accuracy (%)', fontsize=12)
    ax.set_title('HumanEval (164 problems)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylim(0, max(scores) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = f"folder/{output_fname}_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HumanEval results from CSV files with multiple proposal types"
    )
    parser.add_argument(
        "folder", 
        type=str,
    )
    parser.add_argument(
        "output_fname", 
        type=str,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/HumanEval.jsonl",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
    )
    parser.add_argument(
        "--no-common-subset",
        action="store_true",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    fnames = sorted(str(p) for p in folder.glob(args.pattern))
    
    if not fnames:
        raise FileNotFoundError(f"No CSV files found matching {args.pattern} in {folder}")
    
    print(f"Found {len(fnames)} CSV files:")
    for fname in fnames:
        print(f"  - {fname}")
    
    results = he_results(
        fnames, 
        folder,
        args.output_fname, 
        args.data_file,
        use_common_subset=not args.no_common_subset
    )
    
    for method in ['std', 'naive', 'mcmc_priority', 'mcmc_restart', 'mcmc_uniform']:
        if method in results and results[method] is not None:
            pass_at_1 = results[method].get('pass@1', 0.0)
            print(f"{method.upper()}: pass@1 = {pass_at_1:.4f} ({pass_at_1*100:.1f}%)")
        else:
            print(f"{method.upper()}: No results")
    
    if not args.no_plot:
        plot_results(results, args.output_fname)

if __name__ == "__main__":
    main()